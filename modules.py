"""
OctopusNet Modules - The "arms" of the octopus

Each module processes input independently and produces a bottleneck representation.
Modules can be homogeneous (all CNNs) or heterogeneous (CNN + Transformer + LSTM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _rms_norm(x):
    """RMS-normalize a tensor along all non-batch dims.
    Applied to the INPUT of each layer before the conv/linear op.
    Goodness is then measured on the un-renormalized output — not constant.
    Pattern from ASGE (arXiv:2509.12394) and SCFF (Nature Comms 2025).
    """
    rms = x.pow(2).mean(dim=list(range(1, x.dim())), keepdim=True).sqrt() + 1e-8
    return x / rms


class FFLayer(nn.Module):
    """
    A layer trained with Forward-Forward algorithm.
    Based on Hinton (2022) and implementations from:
    - mpezeshki/pytorch_forward_forward
    - loeweX/Forward-Forward
    """

    def __init__(self, in_features, out_features, threshold=2.0, lr=0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Initialize weights
        nn.init.normal_(self.linear.weight, mean=0, std=1/math.sqrt(in_features))
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # Layer normalization (from loeweX)
        x = x / (torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True)) + 1e-8)
        return self.relu(self.linear(x))

    def compute_goodness(self, h):
        """Goodness = sum of squared activations"""
        return torch.sum(h ** 2, dim=-1)

    def ff_loss(self, g_pos, g_neg):
        """
        Forward-Forward loss:
        - Push positive goodness above threshold
        - Push negative goodness below threshold
        """
        loss_pos = torch.log(1 + torch.exp(-(g_pos - self.threshold)))
        loss_neg = torch.log(1 + torch.exp(g_neg - self.threshold))
        return (loss_pos + loss_neg).mean()

    def train_step(self, x_pos, x_neg):
        """Single training step for this layer"""
        h_pos = self.forward(x_pos)
        h_neg = self.forward(x_neg)

        g_pos = self.compute_goodness(h_pos)
        g_neg = self.compute_goodness(h_neg)

        loss = self.ff_loss(g_pos, g_neg)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Detach for next layer (no backprop between layers)
        return h_pos.detach(), h_neg.detach(), loss.item()


def _round_to_multiple(n, m):
    """Round n up to nearest multiple of m."""
    return ((n + m - 1) // m) * m


class CNNModule(nn.Module):
    """
    CNN-based module for OctopusNet.
    Processes images with convolutional layers and outputs bottleneck representation.
    """

    def __init__(self, kernel_size, bottleneck_size=64, in_channels=3,
                 channels=[64, 128, 256], input_size=32, ff_threshold=2.0,
                 adaptive_threshold=False, num_classes=10,
                 use_channel_grouping=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.bottleneck_size = bottleneck_size
        self.num_classes = num_classes
        self.use_channel_grouping = use_channel_grouping

        if use_channel_grouping:
            channels = [_round_to_multiple(c, num_classes) for c in channels]

        self.channels = channels
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size, padding=padding)

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        self.bottleneck = nn.Linear(channels[2] * 4, bottleneck_size)

        self.threshold = ff_threshold
        self.adaptive_threshold = adaptive_threshold
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.feedback = None

        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def _conv_features(self, x):
        """Downsample to module resolution (if needed), then 3 conv layers.
        LayerNorm between layers passes only orientation, not magnitude —
        prevents goodness explosion (Hinton 2022, Scientific Reports 2025).
        Goodness is measured on f3 (un-normalized output of last layer).
        """
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode='bilinear', align_corners=False)
        if self.feedback is not None:
            x = x * self.feedback.view(1, -1, 1, 1)[:, :x.shape[1], :, :]
        f1 = self.relu(self.conv1(x))
        f2 = self.relu(self.conv2(F.layer_norm(f1, f1.shape[1:])))
        f3 = self.relu(self.conv3(F.layer_norm(f2, f2.shape[1:])))
        return f1, f2, f3

    def forward(self, x):
        f1, f2, f3 = self._conv_features(x)
        h = self.pool(f3).view(f3.size(0), -1)
        h = self.bottleneck(h)
        # Normalize only for coordinator/nerve ring — does NOT affect FF goodness
        return h / (h.norm(dim=-1, keepdim=True) + 1e-8)

    def forward_with_features(self, x):
        """Returns (bottleneck, [f1, f2, f3]) for ASGE goodness computation."""
        f1, f2, f3 = self._conv_features(x)
        h = self.pool(f3).view(f3.size(0), -1)
        h = self.bottleneck(h)
        h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        return h, [f1, f2, f3]

    @staticmethod
    def asge_goodness(feature_maps, alpha=0.5):
        """
        Adaptive Spatial Goodness Encoding (ASGE, arxiv 2509.12394).
        Partitions each feature map into spatial patches, computes mean
        squared energy per patch. Richer signal than global sum-of-squares
        because it preserves spatial structure during FF training.

        Args:
            feature_maps: list of (B, C, H, W) tensors from conv layers
            alpha: controls patch count relative to channel depth

        Returns:
            goodness: (B,) scalar per sample — sum of all patch energies
        """
        total = None
        for fm in feature_maps:
            B, C, H, W = fm.shape
            # Patch count: fewer patches for deeper (more channels) layers
            C_max = feature_maps[-1].shape[1]
            P = max(1, min(int(alpha * C_max / C), H, W))

            if P == 1:
                # Single patch = global mean squared activation
                energy = (fm ** 2).mean(dim=[2, 3])  # (B, C)
            else:
                # Reshape into P×P patches, compute energy per patch
                # Trim to make H, W divisible by P
                H_t, W_t = (H // P) * P, (W // P) * P
                fm_t = fm[:, :, :H_t, :W_t]
                # (B, C, P, H//P, P, W//P) → mean over patch dims
                fm_p = fm_t.view(B, C, P, H_t // P, P, W_t // P)
                energy = (fm_p ** 2).mean(dim=[3, 5]).view(B, -1)  # (B, C*P*P)

            layer_goodness = energy.sum(dim=-1)  # (B,)
            total = layer_goodness if total is None else total + layer_goodness

        return total  # (B,)

    def channel_grouping_goodness(self, feature_maps, labels):
        """
        Channel grouping goodness (Ortiz Torres et al., arXiv:2504.21662).
        Single forward pass — no x_pos/x_neg needed.
        Channels split into J groups (one per class). Goodness per group
        computed, then g_pos = group matching true label, g_neg = mean of rest.

        G_{n,j} = (1/S*H*W) * sum(Y[:,j*S:(j+1)*S,:]^2)
        g_pos = G[n, label_n]
        g_neg = mean(G[n, j≠label_n])

        Args:
            feature_maps: list of (B, C, H, W) — must have C % num_classes == 0
            labels: (B,) integer class labels

        Returns:
            g_pos: (B,) goodness for correct class group
            g_neg: (B,) mean goodness for incorrect class groups
        """
        J = self.num_classes
        g_pos_total = None
        g_neg_total = None

        for fm in feature_maps:
            B, C, H, W = fm.shape
            S = C // J  # channels per group

            # (B, J, S, H, W) → mean squared per group
            fm_grouped = fm.view(B, J, S, H, W)
            G = (fm_grouped ** 2).mean(dim=[2, 3, 4])  # (B, J)

            # g_pos: goodness of true class group
            g_pos_layer = G[torch.arange(B), labels]  # (B,)

            # g_neg: mean goodness of all other groups
            mask = torch.ones(B, J, device=fm.device, dtype=torch.bool)
            mask[torch.arange(B), labels] = False
            g_neg_layer = G[mask].view(B, J - 1).mean(dim=1)  # (B,)

            g_pos_total = g_pos_layer if g_pos_total is None else g_pos_total + g_pos_layer
            g_neg_total = g_neg_layer if g_neg_total is None else g_neg_total + g_neg_layer

        return g_pos_total, g_neg_total

    def compute_goodness(self, h):
        """Fallback: global sum of squared activations (Hinton 2022)."""
        return torch.sum(h ** 2, dim=-1)

    def ff_loss(self, g_pos, g_neg):
        """Forward-Forward loss."""
        loss_pos = torch.log(1 + torch.exp(-(g_pos - self.threshold)))
        loss_neg = torch.log(1 + torch.exp(g_neg - self.threshold))
        return (loss_pos + loss_neg).mean()

    def train_ff(self, x_pos, x_neg=None, labels=None):
        """
        Train with FF. Two modes:
        - Standard (x_pos, x_neg): goodness = mean(f3^2) on last conv layer
        - Channel grouping (x_pos=image, labels): single forward pass, goodness by channel group

        WHY mean(f3^2) and not ASGE: ASGE global average dilutes the Fourier
        label signal (only ~6% of spatial area at 32x32). At lower resolutions
        (16,8,4) the signal occupies proportionally more space and global mean
        suffices. Empirically confirmed: all 4 multi-scale modules reach sep>0.05
        in 200 batches with this approach.
        """
        if self.use_channel_grouping and labels is not None:
            _, feats = self.forward_with_features(x_pos)
            g_pos, g_neg = self.channel_grouping_goodness(feats, labels)
        else:
            _, feats_pos = self.forward_with_features(x_pos)
            _, feats_neg = self.forward_with_features(x_neg)
            # Goodness on last conv layer feature map — global mean squared activation
            g_pos = (feats_pos[-1] ** 2).mean(dim=[1, 2, 3])
            g_neg = (feats_neg[-1] ** 2).mean(dim=[1, 2, 3])

        # Adaptive threshold: midpoint between pos and neg goodness each batch
        self.threshold = (g_pos.mean().item() + g_neg.mean().item()) / 2

        loss = self.ff_loss(g_pos, g_neg)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), g_pos.mean().item(), g_neg.mean().item()

    def set_feedback(self, f):
        """Set feedback vector for next forward pass"""
        self.feedback = f


class TransformerModule(nn.Module):
    """
    Mini-Transformer module for heterogeneous OctopusNet.
    Captures global relationships between image patches.
    """

    def __init__(self, bottleneck_size=64, in_channels=3, patch_size=4,
                 embed_dim=64, num_heads=4, num_layers=2, input_size=32,
                 ff_threshold=2.0, adaptive_threshold=False):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.bottleneck_size = bottleneck_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                      kernel_size=patch_size, stride=patch_size)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Bottleneck
        self.bottleneck = nn.Linear(embed_dim, bottleneck_size)

        # FF components
        self.threshold = ff_threshold
        self.adaptive_threshold = adaptive_threshold
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.feedback = None

        # Initialize positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)

    def _embed(self, x):
        """Patch embeddings — used for goodness (pre-transformer, raw energy)."""
        return self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, E)

    def forward(self, x):
        expected = int(self.num_patches ** 0.5) * self.patch_embed.kernel_size[0]
        if x.shape[-1] != expected:
            x = F.interpolate(x, size=(expected, expected),
                              mode='bilinear', align_corners=False)
        h = self._embed(x) + self.pos_embed
        h = self.transformer(h)
        h = h.mean(dim=1)
        h = self.bottleneck(h)
        return h / (h.norm(dim=-1, keepdim=True) + 1e-8)

    def ff_loss(self, g_pos, g_neg):
        loss_pos = torch.log(1 + torch.exp(-(g_pos - self.threshold)))
        loss_neg = torch.log(1 + torch.exp(g_neg - self.threshold))
        return (loss_pos + loss_neg).mean()

    def train_ff(self, x_pos, x_neg):
        # Goodness on patch embeddings — raw conv energy, before transformer
        expected = int(self.num_patches ** 0.5) * self.patch_embed.kernel_size[0]
        if x_pos.shape[-1] != expected:
            x_pos = F.interpolate(x_pos, size=(expected, expected),
                                  mode='bilinear', align_corners=False)
            x_neg = F.interpolate(x_neg, size=(expected, expected),
                                  mode='bilinear', align_corners=False)
        e_pos = self._embed(x_pos)
        e_neg = self._embed(x_neg)
        g_pos = (e_pos ** 2).mean(dim=[1, 2])
        g_neg = (e_neg ** 2).mean(dim=[1, 2])

        if self.adaptive_threshold:
            self.threshold = (g_pos.mean().item() + g_neg.mean().item()) / 2

        loss = self.ff_loss(g_pos, g_neg)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), g_pos.mean().item(), g_neg.mean().item()

    def set_feedback(self, f):
        self.feedback = f


class LSTMModule(nn.Module):
    """
    LSTM module for heterogeneous OctopusNet.
    Processes image as sequence of scanlines to capture sequential patterns.
    """

    def __init__(self, bottleneck_size=64, in_channels=3, hidden_size=128,
                 num_layers=2, input_size=32, ff_threshold=2.0,
                 adaptive_threshold=False):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.input_size = input_size

        # Process each row as a sequence element
        # Input: (B, C, H, W) -> (B, H, C*W)
        self.lstm = nn.LSTM(
            input_size=in_channels * input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Bottleneck (bidirectional = 2x hidden)
        self.bottleneck = nn.Linear(hidden_size * 2, bottleneck_size)

        # FF components
        self.threshold = ff_threshold
        self.adaptive_threshold = adaptive_threshold
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.feedback = None

    def _hidden(self, x):
        """LSTM hidden state with RMS-norm on input sequence."""
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode='bilinear', align_corners=False)
        seq = x.permute(0, 2, 1, 3).reshape(x.size(0), self.input_size, -1)
        seq = _rms_norm(seq)
        out, _ = self.lstm(seq)
        return out[:, -1, :]  # (B, hidden*2)

    def forward(self, x):
        h = self.bottleneck(self._hidden(x))
        return h / (h.norm(dim=-1, keepdim=True) + 1e-8)

    def ff_loss(self, g_pos, g_neg):
        loss_pos = torch.log(1 + torch.exp(-(g_pos - self.threshold)))
        loss_neg = torch.log(1 + torch.exp(g_neg - self.threshold))
        return (loss_pos + loss_neg).mean()

    def train_ff(self, x_pos, x_neg):
        h_pos = self._hidden(x_pos)
        h_neg = self._hidden(x_neg)
        g_pos = (h_pos ** 2).mean(dim=-1)
        g_neg = (h_neg ** 2).mean(dim=-1)

        if self.adaptive_threshold:
            self.threshold = (g_pos.mean().item() + g_neg.mean().item()) / 2

        loss = self.ff_loss(g_pos, g_neg)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), g_pos.mean().item(), g_neg.mean().item()

    def set_feedback(self, f):
        self.feedback = f


def adaptive_kernel(kernel_size, input_size):
    """Clamp kernel to largest odd number <= input_size."""
    max_k = input_size if input_size % 2 == 1 else input_size - 1
    return min(kernel_size, max_k)


def create_modules(config):
    """
    Factory function to create modules based on configuration.

    Args:
        config: OctopusNetConfig

    Returns:
        nn.ModuleList of modules
    """
    modules = nn.ModuleList()

    # Determine input channels and base size based on dataset
    if config.dataset == "mnist" or config.dataset == "fashion_mnist":
        in_channels = 1
        input_size = 28
    else:  # CIFAR
        in_channels = 3
        input_size = 32

    adaptive = getattr(config, 'ff_adaptive_threshold', False)
    use_multiscale = getattr(config, 'use_multiscale', False)
    input_scales = getattr(config, 'input_scales', None)
    use_grouping = getattr(config, 'ff_channel_grouping', False)
    num_classes = getattr(config, 'num_classes', 10)

    if config.homogeneous:
        for i, kernel_size in enumerate(config.kernel_sizes[:config.num_modules]):
            scale = input_scales[i] if use_multiscale and input_scales else input_size
            k = adaptive_kernel(kernel_size, scale) if use_multiscale else kernel_size
            module = CNNModule(
                kernel_size=k,
                bottleneck_size=config.bottleneck_size,
                in_channels=in_channels,
                input_size=scale,
                ff_threshold=config.ff_threshold,
                adaptive_threshold=adaptive,
                num_classes=num_classes,
                use_channel_grouping=use_grouping,
            )
            modules.append(module)
    else:
        hetero_configs = [
            ('cnn',         3, input_scales[0] if use_multiscale and input_scales else input_size),
            ('cnn',         7, input_scales[1] if use_multiscale and input_scales else input_size),
            ('transformer', 0, input_scales[2] if use_multiscale and input_scales else input_size),
            ('lstm',        0, input_scales[3] if use_multiscale and input_scales else input_size),
        ]
        for mtype, ks, scale in hetero_configs:
            if mtype == 'cnn':
                k = adaptive_kernel(ks, scale) if use_multiscale else ks
                modules.append(CNNModule(
                    kernel_size=k, bottleneck_size=config.bottleneck_size,
                    in_channels=in_channels, input_size=scale,
                    ff_threshold=config.ff_threshold, adaptive_threshold=adaptive,
                    num_classes=num_classes, use_channel_grouping=use_grouping,
                ))
            elif mtype == 'transformer':
                modules.append(TransformerModule(
                    bottleneck_size=config.bottleneck_size,
                    in_channels=in_channels, input_size=scale,
                    ff_threshold=config.ff_threshold, adaptive_threshold=adaptive
                ))
            else:  # lstm
                modules.append(LSTMModule(
                    bottleneck_size=config.bottleneck_size,
                    in_channels=in_channels, input_size=scale,
                    ff_threshold=config.ff_threshold, adaptive_threshold=adaptive
                ))

    return modules


class ModuleDecoder(nn.Module):
    """
    Transposed CNN decoder for sleep phase experiments (A16).
    Inverts a CNNModule: bottleneck → image at module's resolution.

    Used in sleep phase training: noise → decoder → dream image → FF negative.
    The .detach() on decoder output is critical — prevents adversarial loop
    where decoder learns to fool the encoder.
    """
    def __init__(self, bottleneck_size=64, out_channels=3, out_size=32):
        super().__init__()
        self.out_size = out_size
        self.fc = nn.Linear(bottleneck_size, 64 * 2 * 2)
        layers = [nn.Unflatten(1, (64, 2, 2))]
        current = 2
        ch = 64
        while current < out_size:
            next_ch = max(ch // 2, out_channels)
            layers += [
                nn.ConvTranspose2d(ch, next_ch, 4, stride=2, padding=1),
                nn.ReLU()
            ]
            ch = next_ch
            current *= 2
        layers += [nn.Conv2d(ch, out_channels, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(self.fc(z))
