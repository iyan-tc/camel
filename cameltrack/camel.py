import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import pytorch_lightning as pl
import transformers

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_metric_learning import distances, losses, reducers

from cameltrack.utils.merge_token_strats import merge_token_strats
from cameltrack.utils.similarity_metrics import similarity_metrics
from cameltrack.utils.assignment_strats import association_strats
from cameltrack.utils.coordinates import norm_coords_strats

log = logging.getLogger(__name__)


@dataclass
class Tracklets:
    """Contains all information of tracklets like a sequence of detections.

    Args:
        features: dict of tensors float32 [B, N, T, F]
        feats_masks: tensor bool [B, N, T] of valid features for padding
        targets: optional, tensor float32 [B, N]

    Attributes:
        feats: dict of tensors float32 [B, N, T, F]
        feats_masks: tensor bool [B, N, T]
        masks: tensor bool [B, N]
        tokens: tensor float32 [B, N, E]
        embs: tensor float32 [B, N, E]
        targets: tensor float32 [B, N]
    """
    def __init__(self, features, feats_masks, targets=None):
        self.feats = features
        self.feats_masks = feats_masks
        self.masks = self.feats_masks.any(dim=-1)
        if targets is not None and len(targets.shape) > 2:
            self.targets = targets[:, :, 0]
        else:
            self.targets = targets


@dataclass
class Detections(Tracklets):
    """Contains all information of detections.

    Args:
        features: dict of tensors float32 [B, N, 1, F]
        feats_masks: tensor bool [B, N, 1] of valid features for padding
        targets: optional, tensor float32 [B, N]

    Attributes:
        feats: dict of tensors float32 [B, N, 1, F]
        feats_masks: tensor bool [B, N, 1]
        masks: tensor bool [B, N]
        tokens: tensor float32 [B, N, E]
        embs: tensor float32 [B, N, E]
        targets: tensor float32 [B, N]
    """
    def __init__(self, features, feats_masks, targets=None):
        assert feats_masks.shape[2] == 1
        super().__init__(features, feats_masks, targets)


class CAMEL(pl.LightningModule):

    def __init__(
            self,
            gaffe: DictConfig,
            temporal_encoders: DictConfig,
            sim_threshold: int = 0.5,
            use_computed_sim_threshold: bool = False,
            optimizer: DictConfig = None,
            merge_token_strat: str = "sum",
            sim_strat: str = "norm_euclidean",
            ass_strat: str = "hungarian_algorithm",
            norm_strat: str = "positive",
    ):
        """
        Initializes the CAMEL tracking model.

        Args:
            gaffe (DictConfig): Configuration for the GAFFE module (Group-Aware Feature Fusion Encoder).
            temporal_encoders (DictConfig): Configurations for the TE (Temporal Encoders).
            sim_threshold (float, optional): Similarity threshold for association.
            use_computed_sim_threshold (bool, optional): Whether to use a computed similarity threshold.
            optimizer (DictConfig, optional): Optimizer configuration for the training.
            merge_token_strat (str, optional): Strategy for merging token representations - for testing purposes.
            sim_strat (str, optional): Similarity metric strategy - for testing purposes.
            ass_strat (str, optional): Association strategy for matching - for testing purposes.
            norm_strat (str, optional): Normalization strategy for coordinates - for testing purposes.

        Raises:
            NotImplementedError: If a provided strategy is not implemented.
        """
        super().__init__()
        self.save_hyperparameters(ignore=[key for key in locals() if "checkpoint_path" in key])

        self.gaffe = instantiate(gaffe)
        # Initializes a Temporal Encoder for each input cue (bbox/keypoints/appearance/...)
        self.temp_encs = nn.ModuleDict({n: instantiate(t, output_dim=gaffe.emb_dim, name=n, _recursive_=False) for n, t in temporal_encoders.items()})

        self.sim_threshold = sim_threshold
        self.use_computed_sim_threshold = use_computed_sim_threshold
        self.computed_sim_threshold = None
        if use_computed_sim_threshold:
            log.warning(f"CAMEL initialized with final_tracking_threshold={sim_threshold} from the configuration file. "
                        "This value will be updated later with an optimized threshold if CAMEL validation is enabled.")

        self.merge_token_strat = "identity" if sim_strat == "default_for_each_token_type" else merge_token_strat
        if self.merge_token_strat in merge_token_strats:
            self.merge = merge_token_strats[self.merge_token_strat]
        else:
            raise NotImplementedError

        self.sim_strat = sim_strat
        if sim_strat in similarity_metrics:
            self.similarity_metric = similarity_metrics[sim_strat]
        else:
            raise NotImplementedError

        if ass_strat in association_strats:
            self.association = association_strats[ass_strat]
        else:
            raise NotImplementedError

        if norm_strat in norm_coords_strats:
            self.norm_coords = norm_coords_strats[norm_strat]
        else:
            raise NotImplementedError

        self.optimizer = optimizer
        # InfoNCE loss
        self.sim_loss = losses.NTXentLoss(distance=distances.CosineSimilarity(), reducer=reducers.AvgNonZeroReducer())


    def training_step(self, batch, batch_idx):
        """
        Single training step in the PyTorch Lightning training loop.
        Returns a dict containing :
            - "loss" (torch.Tensor): The computed loss for the current batch.
            - "dets" (Detections): Processed detections data.
            - "tracks" (Tracklets): Processed tracklets data.
            - "td_sim_matrix" (torch.Tensor): The similarity matrix between tracklets and detections.
        """
        tracks, dets = self.train_val_preprocess(batch)
        tracks, dets, td_sim_matrix = self.forward(tracks, dets)
        loss = self.compute_loss(tracks, dets, td_sim_matrix)
        self.log_loss(loss, "train")
        return {
            "loss": loss,
            "dets": dets,
            "tracks": tracks,
            "td_sim_matrix": td_sim_matrix,
        }

    def validation_step(self, batch, batch_idx):
        """
        Single validation step in the PyTorch Lightning training loop.
        Returns a dict containing :
            - "loss" (torch.Tensor): The computed loss for the current batch.
            - "dets" (Detections): Processed detections data.
            - "tracks" (Tracklets): Processed tracklets data.
            - "td_sim_matrix" (torch.Tensor): The similarity matrix between tracklets and detections.
        """
        tracks, dets = self.train_val_preprocess(batch)
        tracks, dets, td_sim_matrix = self.forward(tracks, dets)
        loss = self.compute_loss(tracks, dets, td_sim_matrix)
        self.log_loss(loss, "val")
        return {
            "loss": loss,
            "tracks": tracks,
            "dets": dets,
            "td_sim_matrix": td_sim_matrix,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Single evaluation step in the PyTorch Lightning training loop.
        Returns a tuple :
            - association_matrix (torch.Tensor): The binary boolean association matrix between tracklets and detections.
            - association_result (list[dict]): Contains the association results, where each dict has:
                - matched_td_indices (list[int]): Indices of matched detections for each tracklet.
                - unmatched_detections (list[int]): Indices of unmatched detections.
                - unmatched_trackers (list[int]): Indices of unmatched tracklets.
            - td_sim_matrix (torch.Tensor): The similarity matrix scores between tracklets and detections.
        """
        tracks, dets = self.predict_preprocess(batch)
        tracks, dets, td_sim_matrix = self.forward(tracks, dets)
        sim_threshold = self.computed_sim_threshold if (self.use_computed_sim_threshold and self.computed_sim_threshold) else self.sim_threshold
        association_matrix, association_result = self.association(td_sim_matrix, tracks.masks, dets.masks, sim_threshold=sim_threshold)
        return association_matrix, association_result, td_sim_matrix

    def forward(self, tracks, dets):
        """
        Performs a forward pass through the CAMEL model.

        Returns:
            tuple: A tuple containing:
                - tracks (Tracklets): Updated tracklets with tokenized and embedded features.
                - dets (Detections): Updated detections with tokenized and embedded features.
                - td_sim_matrix (torch.Tensor): The similarity matrix between tracklets and detections.

        Steps:
            1. Tokenize the input features of tracklets and detections using the linear projections.
            2. Merge the tokenized features into unified token representations using the Temporal Encoders.
            3. Apply the GAFFE module to compute embeddings from the tokens.
            4. Compute the similarity matrix between the tracklet and detection embeddings.
        """
        tracks, dets = self.tokenize(tracks, dets)  # feats -> list(tokens)
        tracks, dets = self.merge(tracks, dets)  # list(tokens) -> tokens
        tracks, dets = self.gaffe(tracks, dets)  # tokens -> embs
        td_sim_matrix = self.similarity(tracks, dets)  # embs -> sim_matrix
        return tracks, dets, td_sim_matrix

    def train_val_preprocess(self, batch):  # TODO merge with predict_preprocess, compute det/trask masks in getitem
        """
        :param batch:
            dict of tensors containing the inputs features and targets of detections and tracklets
        :return:
            dets: Detections - a dataclass wrapper containing batch infos for detections
            tracks: Tracklets - a dataclass wrapper containing batch infos for tracklets
        """
        if self.norm_coords is not None:
            batch = self.norm_coords(batch)
        tracks = Tracklets(batch["track_feats"], ~batch["track_targets"].isnan(), batch["track_targets"])
        dets = Detections(batch["det_feats"], ~batch["det_targets"].isnan(), batch["det_targets"])
        return tracks, dets

    def predict_preprocess(self, batch):
        """
        :param batch:
            dict of tensors containing the inputs features and targets of detections and tracklets
        :return:
            dets: Detections - a dataclass wrapper containing batch infos for detections
            tracks: Tracklets - a dataclass wrapper containing batch infos for tracklets
        """

        if self.norm_coords is not None:
            batch = self.norm_coords(batch)
        tracks = Tracklets(batch["track_feats"], batch["track_masks"],
                           batch["track_targets"] if "track_targets" in batch else None)
        dets = Detections(batch["det_feats"], batch["det_masks"],
                          batch["det_targets"] if "det_targets" in batch else None)
        return tracks, dets

    def tokenize(self, tracks, dets):
        """
        Operate the tokenization step for the different Temporal Encoders.
        :param dets: Detections
        :param tracks: Tracklets
        :return: updated dets and tracks with partial tokens in a dict not merged
        """
        tracks.tokens = {}
        dets.tokens = {}
        for name, temp_enc in self.temp_encs.items():
            tracks.tokens[name] = temp_enc(tracks)
            dets.tokens[name] = temp_enc(dets)
        return tracks, dets

    def similarity(self, tracks, dets):
        """
        Compute the similarity matrix between the detection and tracklet tokens.
        (If tokens is a list of N tensors and not a single tensor, N similarity matrices are computed and averaged.)
        """
        # FIXME similarity_metric should be a list, a different metric could be used for each type of token
        if isinstance(tracks.embs, dict):
            td_sim_matrix = []
            for (tokenizer_name, t), (_, d) in zip(tracks.embs.items(), dets.embs.items()):
                # Each token type has its own default distance (reid = cosine, bbox = iou, etc).
                # Use that default distance for a heuristic that would, for instance, combine IoU with cosine distance.
                if self.sim_strat == "default_for_each_token_type":
                    sm = similarity_metrics[self.temp_encs[tokenizer_name].default_similarity_metric]
                    td_sim_matrix.append(sm(t, tracks.masks, d, dets.masks))
                else:
                    td_sim_matrix.append(self.similarity_metric(t, tracks.masks, d, dets.masks))
            td_sim_matrix = torch.stack(td_sim_matrix).mean(dim=0)
        else:
            td_sim_matrix = self.similarity_metric(tracks.embs, tracks.masks, dets.embs, dets.masks)
        return td_sim_matrix

    def compute_loss(self, tracks, dets, *args):
        """
        :param dets: dataclass
            embs tensor float32 dim [B, D, E]
            confs tensor float32 dim [B, D]
            masks tensor bool dim [B, D]
            targets tensor float32 dim [B, D]
        :param tracks: dataclass
            embs tensor float32 dim [B, T, E]
            masks tensor bool dim [B, T]
            targets tensor float32 dim [B, T]
        :param td_sim_matrix: unused
        :return: sim_loss float32 and cls_loss float32
        """
        if not isinstance(tracks.embs, dict):
            tracks.embs = {"default": tracks.embs}
            dets.embs = {"default": dets.embs}

        n_tokens = len(tracks.embs.keys())
        B = list(tracks.embs.values())[0].shape[0]

        # Initialize loss variables
        sim_loss = torch.zeros((n_tokens, B), dtype=torch.float32, device=self.device)
        mask_sim_loss = torch.zeros((n_tokens, B), dtype=torch.bool, device=self.device)

        for h, token_name in enumerate(tracks.embs.keys()):
            tracks_embs = tracks.embs[token_name]
            dets_embs = dets.embs[token_name]

            for i in range(B):
                masked_track_embs = tracks_embs[i, tracks.masks[i]]
                masked_track_targets = tracks.targets[i, tracks.masks[i]]
                masked_det_embs = dets_embs[i, dets.masks[i]]
                masked_det_targets = dets.targets[i, dets.masks[i]]

                if ((len(masked_det_embs) != 0 or len(masked_track_embs) != 0) or (len(masked_det_embs) > 1 or len(masked_track_embs) > 1)):
                    mask_sim_loss[h, i] = True

                    # Compute embeddings' loss on all tracks/detections (track_ids >= 0)
                    valid_tracks = masked_track_targets >= 0
                    valid_dets = masked_det_targets >= 0
                    embeddings = torch.cat([masked_track_embs[valid_tracks],
                                            masked_det_embs[valid_dets]], dim=0)
                    labels = torch.cat([masked_track_targets[valid_tracks],
                                        masked_det_targets[valid_dets]], dim=0)
                    sim_loss[h, i] = self.sim_loss(embeddings, labels)

        # Compute mean losses over valid items in the batch
        sim_loss = sim_loss[mask_sim_loss].mean()
        # Handle NaN values
        sim_loss = sim_loss.nan_to_num(0)

        return sim_loss

    def log_loss(self, loss, step):
        # pytorch-lightning way to log
        self.log_dict(
            {f"{step}/loss": loss},
            on_epoch=True,
            on_step="train" == step,
            prog_bar="train" == step,
            logger=True,
        )

    def configure_optimizers(self):
        # pytorch-lightning way to init the optimizer and scheduler
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optimizer.init_lr, weight_decay=self.optimizer.weight_decay)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.trainer.estimated_stepping_batches // 20,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_save_checkpoint(self, checkpoint):
        # Add custom attributes to the checkpoint dictionary
        checkpoint['computed_sim_threshold'] = self.computed_sim_threshold

    def on_load_checkpoint(self, checkpoint):
        # Load custom attributes from the checkpoint dictionary
        self.computed_sim_threshold = checkpoint.get('computed_sim_threshold', None)
        self.on_validation_end()
