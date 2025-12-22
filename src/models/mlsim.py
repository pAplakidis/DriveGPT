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
      tokens_only = False,
      device=torch.device("cpu")
    ):
    super().__init__()
    self.config = config
    self.inference = inference
    self.tokens_only = tokens_only
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
    self.token_head = nn.Linear(self.config.embedding_dim, self.vqvae_cfg.n_embeddings)

    self.cache = None

  def forward(self, x, actions, targets=None):
    # TODO: load from self.cache if self.inference

    if self.tokens_only:
      B, T, H_e, W_e = x.shape
      min_encoding_indices = x.view(B*T, H_e, W_e)
    else:
      B, T, C, H, W = x.shape
      flat = x.reshape(B * T, C, H, W)
      with torch.no_grad():
        embedding_loss, z_q, perplexity, min_encodings, min_encoding_indices = self.image_tokenizer(flat)
      H_e, W_e = z_q.shape[2:]  # latent spatial size
      min_encoding_indices = min_encoding_indices.view(B*T, H_e, W_e)  # now correct shape

    image_tokens = min_encoding_indices.view(B, T, H_e, W_e)  # (B, T, H_e, W_e)

    # z_q[b, :, h, w] == embedding.weight[ indices[b, h, w] ]
    # image_tokens = z_q.reshape(B, T, *z_q.shape[1:])
    # if self.inference: self.cache = image_tokens

    # map token indices to embeddings for transformer
    token_embeddings = F.embedding(image_tokens, self.vqvae.vector_quantization.embedding.weight) # (B, T, H_e, W_e, C_e)
    C_e = token_embeddings.shape[-1]

    action_tokens = self.action_embeddings(actions)
    D = action_tokens.shape[-1]

    seq_tokens = token_embeddings.permute(0, 1, 3, 4, 2).reshape(B, T * H_e * W_e, C_e)
    action_tokens_exp = action_tokens.unsqueeze(2).expand(-1, -1, H_e * W_e, D).reshape(B, T * H_e * W_e, D)
    seq_tokens = seq_tokens + action_tokens_exp

    out = self.dynamics_transformer(tgt=seq_tokens, memory=seq_tokens)
    # next_image_emb = out[:, -H_e*W_e:, :].reshape(B, H_e, W_e, C_e).permute(0, 3, 1, 2)
    logits = self.token_head(out[:, -H_e*W_e:, :])                  # (B, H_e*W_e, embedding_dim)
    logits = logits.view(B, H_e, W_e, self.vqvae_cfg.n_embeddings) # (B, H_e, W_e, n_embeddings)

    # TODO: for inference/reconstruction (change this)
    if not self.tokens_only:
      pred_tokens = logits.argmax(dim=-1)  # (B, H_e, W_e)
      embedding = self.vqvae.vector_quantization.embedding
      z_q = embedding(pred_tokens)                # (B, H_e, W_e, C_e)
      z_q = z_q.permute(0, 3, 1, 2).contiguous()  # (B, C_e, H_e, W_e)

      if self.inference:
        # self.cache = torch.cat((token_embeddings[:, 1:], next_image_emb.unsqueeze(1)), dim=1)
        return self.image_decoder(z_q)
      return z_q


    if targets is None:
      loss = None
    else:
      loss = F.cross_entropy(
        logits.view(-1, self.vqvae_cfg.n_embeddings),
        targets.view(-1)
      )

    return logits, loss

  def generate(self, tokens, max_new_tokens):
    pass
