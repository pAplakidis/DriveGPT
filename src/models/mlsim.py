import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from image_tokenizer.config import VQVAEConfig
from image_tokenizer.vqvae.models.vqvae import VQVAE


@dataclass
class MLSimConfig:
  vqvae_path: str = "image_tokenizer/checkpoints/compression_1280_bits/vqvae_best.pt"
  freeze_backbone: bool = True
  n_actions: int = 2
  sequence_size: int = N_FRAMES
  embedding_dim: int = 32
  filters_1d: int = 32
  transformer_heads: int = 8
  transformer_layers: int = 6
  transformer_dropout: int = 0.1
  activation: str = "relu"
  linear_dropout: int = 0.1


class MLSim(nn.Module):
  def __init__(
      self,
      config: MLSimConfig,
      inference=False,
      device=torch.device("cpu")
    ):
    super().__init__()
    self.config = config
    self.inference = inference
    self.device = device

    self.vqvae_cfg = VQVAEConfig()
    self.vqvae = VQVAE(
      self.vqvae_cfg.n_hiddens,
      self.vqvae_cfg.n_residual_hiddens,
      self.vqvae_cfg.n_residual_layers,
      self.vqvae_cfg.n_embeddings,
      self.vqvae_cfg.embedding_dim,
      self.vqvae_cfg.beta
    )
    if os.path.exists(self.config.vqvae_path):
      checkpoint = torch.load(self.config.vqvae_path, map_location=self.device, weights_only=False)
      self.vqvae.load_state_dict(checkpoint["model"])
    if self.config.freeze_backbone:
      for p in self.vqvae.parameters(): # TODO: unfreeze decoder in order to train alongside RNN state
        p.requires_grad = False

    self.image_tokenizer = nn.Sequential(
      self.vqvae.encoder,
      self.vqvae.pre_quantization_conv,
      self.vqvae.vector_quantization
    )
    self.image_decoder = self.vqvae.decoder # TODO: RNN state for smooth video

    # TODO: use Conv1D (?)
    self.action_embeddings = nn.Sequential(
      nn.Linear(self.config.n_actions, self.config.embedding_dim),
      nn.LayerNorm(self.config.embedding_dim)
    )

    # GPT-like transformer decoder
    self.transformer_decoder_layer = nn.TransformerDecoderLayer(
      d_model=self.config.embedding_dim,
      nhead=self.config.transformer_heads,
      dim_feedforward=128,
      dropout=self.config.transformer_dropout,
      activation=self.config.activation,
      batch_first=True
    )
    self.dynamics_transformer = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.config.transformer_layers)

    self.cache = None

  def forward(self, images, actions):
    if self.inference and images is None and self.cache:
      image_tokens, action_tokens = self.cache
    else:
      BS, T, C, H, W = images.shape
      flat = images.reshape(BS * T, C, H, W)
      embedding_loss, z_q, perplexity, min_encodings, min_encoding_indices = self.image_tokenizer(flat)
      image_tokens = z_q.reshape(BS, T, *z_q.shape[1:])
      if self.inference: self.cache = image_tokens

    action_tokens = self.action_embeddings(actions)

    _, _, C_e, H_e, W_e = image_tokens.shape
    _, _, D = action_tokens.shape

    seq_tokens = image_tokens.permute(0, 1, 3, 4, 2).reshape(BS, T * H_e * W_e, C_e)
    action_tokens_exp = action_tokens.unsqueeze(2).expand(-1, -1, H_e * W_e, C_e).reshape(BS, T * H_e * W_e, D)
    seq_tokens += action_tokens_exp

    out = self.dynamics_transformer(tgt=seq_tokens, memory=seq_tokens)
    next_image_tokens = out[:, -H_e*W_e:, :].reshape(BS, H_e, W_e, C_e).permute(0, 3, 1, 2)

    if self.inference:
      self.cache = torch.cat((image_tokens[:, 1:], next_image_tokens.unsqueeze(0)), dim=1)
      return self.image_decoder(next_image_tokens)

    # TODO: train using MSE(z_t+1, Z_t+1)
    return next_image_tokens
