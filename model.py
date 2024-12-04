#!/usr/bin/env python3
import torch
import torch.nn as nn

from config import *

class Encoder(nn.Module):
  def __init__(self, df_dim, ch, rows, cols, z_dim):
    super(Encoder, self).__init__()
    self.df_dim = df_dim
    self.z_dim = z_dim

    self.conv_layers = nn.Sequential(
      nn.Conv2d(in_channels=ch, out_channels=df_dim, kernel_size=5, stride=2, padding=2),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(in_channels=df_dim, out_channels=df_dim * 2, kernel_size=5, stride=2, padding=2),
      nn.BatchNorm2d(df_dim * 2, eps=1e-5),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(in_channels=df_dim * 2, out_channels=df_dim * 4, kernel_size=5, stride=2, padding=2),
      nn.BatchNorm2d(df_dim * 4, eps=1e-5),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(in_channels=df_dim * 4, out_channels=df_dim * 8, kernel_size=5, stride=2, padding=2),
      nn.BatchNorm2d(df_dim * 8, eps=1e-5),
      nn.LeakyReLU(0.2, inplace=True),
      
      nn.Flatten()
    )

    # self.fc_mean = nn.Linear(df_dim * 8 * rows[-1] * cols[-1], z_dim) # FIXME: input shape
    # self.fc_logsigma = nn.Linear(df_dim * 8 * rows[-1] * cols[-1], z_dim) # FIXME: input shape
    self.fc_mean = nn.Linear(25600, z_dim)
    self.fc_logsigma = nn.Linear(25600, z_dim)

  def forward(self, x):
    x = self.conv_layers(x)
    mean = self.fc_mean(x)
    logsigma = torch.tanh(self.fc_logsigma(x))
    return mean, logsigma

class Generator(nn.Module):
  def __init__(self, gf_dim, ch, rows, cols, z_dim):
    super(Generator, self).__init__()

    self.fc = nn.Sequential(
      nn.Linear(z_dim, gf_dim * 8 * rows[0] * cols[0]),
      nn.BatchNorm1d(gf_dim * 8 * rows[0] * cols[0], eps=1e-5),
      nn.ReLU(True)
    )
        
    self.deconv_layers = nn.Sequential(
      nn.ConvTranspose2d(in_channels=gf_dim * 8, out_channels=gf_dim * 4, kernel_size=5, stride=2, padding=2, output_padding=1),
      nn.BatchNorm2d(gf_dim * 4, eps=1e-5),
      nn.ReLU(True),

      nn.ConvTranspose2d(in_channels=gf_dim * 4, out_channels=gf_dim * 2, kernel_size=5, stride=2, padding=2, output_padding=1),
      nn.BatchNorm2d(gf_dim * 2, eps=1e-5),
      nn.ReLU(True),

      nn.ConvTranspose2d(in_channels=gf_dim * 2, out_channels=gf_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
      nn.BatchNorm2d(gf_dim, eps=1e-5),
      nn.ReLU(True),

      nn.ConvTranspose2d(in_channels=gf_dim, out_channels=ch, kernel_size=5, stride=2, padding=2, output_padding=1),
      nn.Tanh()
    )

    self.rows = rows[0]
    self.cols = cols[0]
    self.gf_dim = gf_dim

  def forward(self, x):
    print(x.shape)
    x = self.fc(x).view(-1, self.gf_dim * 8, self.rows, self.cols)
    print(x.shape)
    x = self.deconv_layers(x)
    return x

class Autoencoder(nn.Module):
  def __init__(self, rows, cols, z_dim, df_dim=64, gf_dim=64, ):
    super(Autoencoder, self).__init__()

    self.encoder = Encoder(df_dim, ch, rows, cols, z_dim)
    self.generator = Generator(gf_dim, ch, rows, cols, z_dim)

  def forward(self, x):
    self.mean, self.logsigma = self.encoder(x)

    # z = mean + exp(logsigma) * ϵ
    std = torch.exp(0.5 * self.logsigma)
    epsilon = torch.randn_like(std)
    z = self.mean + std * epsilon

    x = self.generator(z)
    return x

class Discriminator(nn.Module):
  def __init__(self, df_dim, ch, rows, cols):
    super(Discriminator, self).__init__()

    self.conv_layers = nn.Sequential(
      nn.Conv2d(in_channels=ch, out_channels=df_dim, kernel_size=5, stride=2, padding=2),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(in_channels=df_dim, out_channels=df_dim * 2, kernel_size=5, stride=2, padding=2),
      nn.BatchNorm2d(df_dim * 2, eps=1e-5),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(in_channels=df_dim * 2, out_channels=df_dim * 4, kernel_size=5, stride=2, padding=2),
      nn.BatchNorm2d(df_dim * 4, eps=1e-5),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(in_channels=df_dim * 4, out_channels=df_dim * 8, kernel_size=5, stride=2, padding=2),
      nn.BatchNorm2d(df_dim * 8, eps=1e-5),
      nn.LeakyReLU(0.2, inplace=True),
      
      nn.Flatten(),
      nn.Linear((rows // 16) * (cols // 16) * df_dim * 8, 1)
    )

  def forward(self, x):
    output = self.conv_layers(x)
    return output, x


def save_model(model, path):
  pass

def load_model(model, path):
  pass

def torch_to_onnx(model, path):
  pass


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("[+] Using device:", device)

  autoencoder = Autoencoder(rows, cols, z_dim).to(device)
  X = torch.ones((4, 3, 80, 160), dtype=torch.float).to(device)
  out = autoencoder(X)
  print(out)
  print(out.shape)

  # TODO: test discriminator
  # Img = D.input
  # G_train = G(Z)
  # E_mean, E_logsigma = E(Img)
  # G_dec = G(E_mean + Z2 * E_logsigma)
  # D_fake, F_fake = D(G_train)
  # D_dec_fake, F_dec_fake = D(G_dec)
  # D_legit, F_legit = D(Img)
