# LUMA-unbias

An attempt to aliviate the bias present in [LUMA](https://arxiv.org/abs/2406.09864)'s textual moduality.

## Text as a latent approach

The idea is for the generated captions to behave as some kind of latent space (as of that of autoencoders), this latent space must have as much information as possible so that we can reproduce the original image, by avoiding the use of non-factual signals. I state that because of this it could also help in avoiding the inclusion of biases.

1. Generate the captions using a VLM (eg. Paligamma).
2. Use the captions to generate an image, using some kind of diffusion model (eg. Stable diffusion).
3. Use a vision encoder (eg. a ViT) to generate embeddings of both the reference image and the generated image.
4. Compare the generated embeddings using some similarity metric.
5. Pick only the most similar items.

The major issue of this approach is that there are confounding factors, low similarity could indicate any of:

1. Poor VLM descriptions
2. Poor image generation
3. Poor choice of a similarity metric

## Testing with some similarity metrics

### Cosine similarity (ViT Embeds) (Higher Better)

The results show that the similarity scores between generated images and their ground truth is lower! than that of ground truth with images from other classes

![vit_cosim_in_out_sim](https://github.com/user-attachments/assets/75cc6a6e-92d2-404c-9a06-f6988cfc3d48)

This is more likely a result of the poor choice of a similarity metric, as it's unlikely that images from other classes would be more similar the generated ones.

### Lpips (AlexNet) (Lower better)

Surprisingly the Lpips metric seems to be worse at distinguishing between in and out class images, we should be a fundamental basis for better asessing the similarities of images.

The inverstigation must continue by comparing similarity scores between generated images and their ground truth.

![download](https://github.com/user-attachments/assets/ef046455-4332-4968-ac2d-62401d0fc552)

## Next steps

- Exploring other similarity metrics like:
  - LPIPS (Learned Perceptual Image Patch Similarity) of generated images and their ground truth
  - SSIM for structural similarity
  
## Reference

```bibtex

@article{bezirganyan2024lumabenchmarkdatasetlearning,
      title={LUMA: A Benchmark Dataset for Learning from Uncertain and Multimodal Data}, 
      author={Grigor Bezirganyan and Sana Sellami and Laure Berti-Équille and Sébastien Fournier},
      year={2024},
      eprint={2406.09864},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.09864}, 
}
```
