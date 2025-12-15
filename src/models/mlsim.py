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
  embedding_dim: int = 32
  transformer_heads: int = 8
  transformer_layers: int = 6
  transformer_dropout: int = 0.1
  activation: str = "relu"


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
    for p in self.vqvae.parameters():
      p.requires_grad = False

    self.image_tokenizer = nn.Sequential(
      self.vqvae.encoder,
      self.vqvae.pre_quantization_conv,
      self.vqvae.vector_quantization
    )
    self.image_decoder = self.vqvae.decoder

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

  def forward(self, images, actions):
    BS, T, C, H, W = images.shape
    flat = images.reshape(BS * T, C, H, W)
    embedding_loss, z_q, perplexity, min_encodings, min_encoding_indices = self.image_tokenizer(flat)
    # TODO: since this model is autoregressive, save the sequence of tokens at inference
    image_tokens = z_q.reshape(BS, T, *z_q.shape[1:])

    # TODO: concat poses (and later actions) with tokens per image

    _, _, C_e, H_e, W_e = image_tokens.shape
    seq_tokens = image_tokens.permute(0, 1, 3, 4, 2).reshape(BS, T * H_e * W_e, C_e)
    out = self.dynamics_transformer(tgt=seq_tokens, memory=seq_tokens)
    next_image_tokens = out[:, -H_e*W_e:, :].reshape(BS, H_e, W_e, C_e).permute(0, 3, 1, 2)

    if self.inference:
      return self.image_decoder(next_image_tokens)
    
    # TODO: train using MSE(z_t+1, Z_t+1)
    return next_image_tokens
