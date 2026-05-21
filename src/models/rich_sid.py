"""Rich-SID variant of the visual world model.

Keeps the existing :class:`VisionEncoder` / :class:`VisualWorldModel` byte-untouched
and adds a parallel encoder that exposes the intermediate ``(B, 64, 16, 16)``
backbone feature map for a downstream 3D-CNN system-identification (SID) branch.

The hypothesis being tested: PAVER-LH's per-frame 64-dim bottleneck loses
motion-relevant information before the GRU SID ever sees it. Forking the
encoder mid-stack and handing pre-bottleneck spatial features to the SID
branch may close the gap to HGN.

This module only provides the foundation (encoder + VWM wrapper). A
downstream predictor that consumes the rich features will be built on top
of :class:`RichSIDVisualWorldModel` in a follow-up task.
"""

import torch
import torch.nn as nn

from src.models.visual import VisionDecoder, _ResBlock


class RichSIDVisionEncoder(nn.Module):
    """Per-frame vision encoder with a rich-feature fork point.

    Architecture is identical to :class:`src.models.visual.VisionEncoder`
    (same conv stack, same MLP head) so that downstream comparisons against
    PAVER-LH are capacity-matched. The only difference is that the conv
    stack is split in two:

    * ``self.backbone``       : 64×64 → 32×32 → 16×16, ending with a ``_ResBlock``
                                at 16×16. Output shape ``(B, 64, 16, 16)``.
    * ``self.per_frame_head`` : 16×16 → 8×8, ending with a ``_ResBlock`` at 8×8.
                                Output shape ``(B, 64, 8, 8)``.
    * ``self.mlp``            : flatten (B, 4096) → ``hidden_channels`` →
                                ``latent_channels``.

    The fork at ``(B, 64, 16, 16)`` is what a 3D-CNN SID branch consumes.

    Note: ``encoder_frames`` is intentionally not a parameter — temporal
    mixing happens in the SID 3D CNN, not via channel-concatenation at the
    encoder. Input is always a single frame ``(B, C, H, W)``.
    """

    def __init__(self, channels=3, latent_channels=64, hidden_channels=512):
        super().__init__()
        self.latent_channels = latent_channels

        # Stages 1-3 of the original VisionEncoder conv stack: 64×64 → 16×16.
        self.backbone = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1, 1),  # 64×64
            nn.LeakyReLU(0.2),
            _ResBlock(64),                     # 64×64
            nn.Conv2d(64, 64, 4, 2, 1),        # 64→32
            nn.LeakyReLU(0.2),
            _ResBlock(64),                     # 32×32
            nn.Conv2d(64, 64, 4, 2, 1),        # 32→16
            nn.LeakyReLU(0.2),
            _ResBlock(64),                     # 16×16  ← FORK POINT
        )

        # Stage 4 of the original conv stack: 16×16 → 8×8.
        self.per_frame_head = nn.Sequential(
            nn.Conv2d(64, 64, 4, 2, 1),        # 16→8
            nn.LeakyReLU(0.2),
            _ResBlock(64),                     # 8×8
        )

        # Same MLP head as VisionEncoder.
        self.mlp = nn.Sequential(
            nn.Linear(64 * 8 * 8, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels, latent_channels),
        )

    def encode_features(self, x):
        """Encode an image to the (B, 64, 16, 16) backbone feature map.

        Args:
            x: ``(B, C, H, W)`` image tensor.
        Returns:
            feats: ``(B, 64, 16, 16)`` pre-bottleneck spatial features.
        """
        return self.backbone(x)

    def encode_from_features(self, feats):
        """Reduce backbone features to the per-frame latent.

        Args:
            feats: ``(B, 64, 16, 16)`` backbone feature map.
        Returns:
            z: ``(B, latent_channels)`` per-frame latent.
        """
        return self.mlp(self.per_frame_head(feats).flatten(1))

    def forward(self, x):
        return self.encode_from_features(self.encode_features(x))


class RichSIDVisualWorldModel(nn.Module):
    """Visual world model that exposes rich pre-bottleneck features to the SID branch.

    Mirrors the contract of :class:`src.models.visual.VisualWorldModel`
    (``encode_sequence`` → ``(B, T, D)``, ``decode``, BatchNorm projector) so
    that downstream training and evaluation scaffolding can swap models with
    minimal changes. Adds :meth:`encode_features_sequence` which returns the
    full ``(B, T, 64, 16, 16)`` feature tensor for the 3D-CNN SID branch.

    ``encoder_frames`` is fixed at 1: temporal mixing for system identification
    happens in the SID 3D CNN that consumes the feature sequence, not via
    channel-concatenation at the encoder. Constructing with any other value
    raises ``ValueError``.
    """

    def __init__(
        self,
        predictor,
        latent_channels=64,
        hidden_channels=512,
        context_length=3,
        pred_length=1,
        channels=3,
        observation_dt=0.1,
        infer_context_length=None,
        encoder_frames=1,
        **kwargs,
    ):
        super().__init__()

        if encoder_frames != 1:
            raise ValueError(
                f"RichSIDVisualWorldModel requires encoder_frames=1 (got {encoder_frames}). "
                "Temporal mixing happens in the SID 3D CNN."
            )

        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.context_length = context_length
        self.pred_length = pred_length
        self.infer_context_length = (
            infer_context_length if infer_context_length is not None else context_length
        )
        self.observation_dt = observation_dt
        self.encoder_frames = 1
        self.channels = channels

        self.encoder = RichSIDVisionEncoder(
            channels=channels,
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
        )
        # Decoder is identical to VisualWorldModel's — sees the full latent z.
        self.decoder = VisionDecoder(
            channels=channels,
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
        )
        self.predictor = predictor

        # BatchNorm projector for JEPA / SIGReg compatibility — matches
        # VisualWorldModel exactly so training stays apples-to-apples.
        self.encoder_projector = nn.Sequential(
            nn.Linear(latent_channels, latent_channels),
            nn.BatchNorm1d(latent_channels),
        )

    def encode(self, images):
        """Encode a single batch of frames to per-frame latents.

        Args:
            images: ``(B, C, H, W)``.
        Returns:
            z: ``(B, latent_channels)``. The encoder_projector is NOT applied
            here — matches :meth:`VisualWorldModel.encode`'s contract. Use
            :meth:`encode_sequence` for the projector-applied path.
        """
        return self.encoder(images)

    def encode_sequence(self, images):
        """Encode a frame sequence per-frame, with the BatchNorm projector.

        Args:
            images: ``(B, T, C, H, W)``.
        Returns:
            z_seq: ``(B, T, latent_channels)`` after the BatchNorm projector.
        """
        B, T, C, H, W = images.shape
        flat = images.reshape(B * T, C, H, W)
        z = self.encode(flat)                # (B*T, latent_channels)
        z = self.encoder_projector(z)        # (B*T, latent_channels)
        D = z.shape[-1]
        return z.reshape(B, T, D)

    def encode_features_sequence(self, images):
        """Encode a frame sequence to the (B, T, 64, 16, 16) backbone feature map.

        The encoder_projector is NOT applied — it's a per-frame latent-space
        operation, not meaningful on spatial features. This is the entry
        point for the downstream 3D-CNN SID branch.

        Args:
            images: ``(B, T, C, H, W)``.
        Returns:
            feat_seq: ``(B, T, 64, 16, 16)``.
        """
        B, T, C, H, W = images.shape
        flat = images.reshape(B * T, C, H, W)
        feats = self.encoder.encode_features(flat)  # (B*T, 64, 16, 16)
        _, Cf, Hf, Wf = feats.shape
        return feats.reshape(B, T, Cf, Hf, Wf)

    def decode(self, z):
        """Decode a latent to images.

        Args:
            z: ``(B, D)`` or ``(B*T, D)``.
        Returns:
            images: ``(B, C, H, W)``.
        """
        return self.decoder(z)

    def encoder_parameters(self):
        yield from self.encoder.parameters()

    def decoder_parameters(self):
        yield from self.decoder.parameters()

    def predictor_parameters(self):
        yield from self.predictor.parameters()
