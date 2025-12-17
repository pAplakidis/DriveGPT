# DriveGPT

A toy implementation of using VQVAE

## Resources

- [Learning a Driving Simulator](https://arxiv.org/abs/1608.01230)
- [commavq](https://github.com/commaai/commavq)
- [Learning to Drive from a World Model](https://arxiv.org/abs/2504.19077)
- [VQVAE](https://arxiv.org/abs/1711.00937)
- [GAIA-1](https://arxiv.org/abs/2309.17080)

### TODO

- Too slow to train without large compute
  - pre-tokenize dataset images once (offline)
  - do not run vqvae (self.model.image_tokenizer) in trainer.train_step()
  - reduce transformer token count (currently T × H_e × W_e tokens) => pool spatially OR flatten spatial>MLP>smaller token
  - Sanity check with profiler
  ```python
  torch.cuda.synchronize()
  start = time.time()
  ...
  torch.cuda.synchronize()
  print("Batch time:", time.time() - start)
  ```
- Train
- Inference App

Need to go from this:

```
pixels ──▶ tokenizer ──▶ tokens
        └──────────────▶ tokenizer AGAIN
tokens ──▶ transformer (1000-token seq)
```

to this:

```
pixels ──▶ tokenizer (ONCE, offline)
tokens ──▶ transformer dynamics (TRAINED)
tokens ──▶ decoder (optional, eval only)
```
