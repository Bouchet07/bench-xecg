import torch
import torch.nn as nn
from .base_model import BaseModel


class MyTestModel(BaseModel):
    def __init__(self, config, feature_classification=False, sleep_apnea=False):
        super().__init__()
        self.patch_size = config.patch_size
        self.sampling_freq = config.sampling_freq
        self.linear_probing = config.linear_probing
        self.feature_classification = feature_classification
        self.sleep_apnea = sleep_apnea
        self.context_size = config.context_size or 0
        self.window_size = config.window_size or 60

        # --- 1. Pre-trained Backbone ---
        embed_dim = 64  # encoder's output embedding size
        self.encoder = nn.Linear(len(config.leads) * self.patch_size, embed_dim)

        # --- 2. Downstream Task Head (MUST be named self.head) ---
        self.head = nn.Linear(embed_dim, config.num_classes)

    def encode_patches(self, x):
        """
        model's forward pass
        Input `x`: (Batch, Time, Leads)
        Must return patch embeddings of shape: (Batch, Num_Patches, embed_dim)
        """
        B, T, C = x.shape
        num_patches = T // self.patch_size
        x = x[:, :num_patches * self.patch_size, :].reshape(B, num_patches, self.patch_size * C)
        return self.encoder(x)

    def forward(self, x, age=None, gender=None):
        # 1. Extract patch features (with no_grad if linear probing)
        if self.linear_probing:
            with torch.no_grad():
                features = self.encode_patches(x)  # (B, Num_Patches, embed_dim)
        else:
            features = self.encode_patches(x)      # (B, Num_Patches, embed_dim)

        # 2. Task Mode A: Sleep Apnea (slice out surrounding context patches, then pool)
        if self.sleep_apnea:
            if self.context_size > 0:
                ctx_patches = (self.context_size * self.sampling_freq) // self.patch_size // 2
                win_patches = (self.window_size * self.sampling_freq) // self.patch_size
                features = features[:, ctx_patches : ctx_patches + win_patches, :]
            out = features.mean(dim=1)             # (B, embed_dim)
            return self.head(out)                  # (B, num_classes)

        # 3. Task Mode B: Beat / R-peak tasks (MIT-BIH cls, MIT-BIH R-peaks, Exercise R-peaks)
        if self.feature_classification:
            return self.head(features)             # (B, Num_Patches, num_classes)

        # 4. Task Mode C: Standard signal-level tasks (PTB-XL, CPSC2018, Age, Lab, Survival, PPG)
        out = features.mean(dim=1)                 # (B, embed_dim)
        return self.head(out)                      # (B, num_classes)