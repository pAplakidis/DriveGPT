# DriveGPT

A toy implementation of using VQVAE

## Resources

- [Learning a Driving Simulator](https://arxiv.org/abs/1608.01230)
- [Learning to Drive from a World Model](https://arxiv.org/abs/2504.19077)
- [VQVAE](https://arxiv.org/abs/1711.00937)
- [commavq](https://github.com/commaai/commavq)

### TODO

* Pretrain Autoencoder (VQVAE or DiT)
  - Loss (MAE(L1), MSE(L2), perceptual loss, combination: L = a * L_recon(L1?) + b * L_ssim + l * L_latent_reg)
  - Evaluation Metrics (structural similarity index (SSIM), LPIPS, FID, IOU, pixel accuracy)
  - Check out paper + latent space quality metrics
  - Prepare full stack of train/eval/repeat with simple model
  - Implement Better/More complex VQVAE + scale
  - Train
* Train world model
