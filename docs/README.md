<div style="display: flex; align-items: center;">
  <img src="assets/images/prisma.jpg" alt="Vit Prisma Logo" style="margin-right: 10px;"/>
</div>


# Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video
Prisma contains code for vision and video mechanistic interpretability, including activation caching and SAE training. We support a variety of vision/video models from Huggingface and OpenCLIP. This library was originally made by [Sonia Joseph](https://github.com/soniajoseph) (see a full list of contributors [here](#Contributors)).

Mechanistic interpretability is broadly split into two parts: circuit-analysis and sparse autoencoders (SAEs). Circuit-analysis finds the causal links between internal components of the model and primarily relies on activation caching. SAEs are like a more fine-grained "primitive" that you can use to examine intermediate activations. Prisma has the infrastructure to do both.

We also include a suite of [open source SAEs for all layers of CLIP and DINO](https://github.com/Prisma-Multimodal/ViT-Prisma/blob/main/docs/sae_table.md), including transcoders for all layers of CLIP, that you can download from Huggingface.

**For more details, check out our whitepaper [Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video](https://arxiv.org/abs/2504.19475).** Also, check out the original Less Wrong post [here](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic).

**Table of Contents**
- [Installation](#Installation)
- [Models Supported](#Models-Supported)
- [Vision SAE Demo Notebooks](#SAE-Demo-Notebooks)
- [Vision SAE Pretrained Weights](#Suite-of-Pretrained-Vision-SAE-Weights)
  - [CLIP Vanilla SAEs (All Patches)](#clip-vanilla-saes-all-patches)
  - [DINO](#dino-vanilla-all-patches)
  - [CLIP transcoders](#clip-transcoders)
  - For more, see [the full table](/docs/sae_table.md)
- [Basic Mechanistic Interpretability](#Basic-Mechanistic-Interpretability)
- [Contributors](#Contributors)
- [Citation](#Citation)

# Installation

We recommend installing from source:

```
git clone https://github.com/soniajoseph/ViT-Prisma
cd ViT-Prisma
pip install -e .
```

# Models Supported
We support most vision/video transformers loaded from OpenCLIP and Huggingface, including ViTs, CLIP, DINO, and JEPA, with a few exceptions (e.g. if the architecture is substantially different).

For a list of model names, check out our model registry [here](https://github.com/Prisma-Multimodal/ViT-Prisma/blob/main/src/vit_prisma/models/model_config_registry.py).

To load a model:
```
from vit_prisma.models.model_loader import load_hooked_model

model_name = "open-clip:laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K
model = load_hooked_model(model_name)
model.to('cuda') # Move to cuda if available
```
# SAE Pretrained Weights, Training, and Evaluation Code

## SAE Demo Notebooks
Here are notebooks to load, train, and evaluate SAEs on the supported models:
* [SAE loading notebook](https://github.com/Prisma-Multimodal/ViT-Prisma/blob/main/demos/1_Load_SAE.ipynb)
* [SAE training notebook](https://github.com/Prisma-Multimodal/ViT-Prisma/blob/main/demos/2_Train_SAE.ipynb)
* [SAE evaluation notebook](https://github.com/Prisma-Multimodal/ViT-Prisma/blob/main/demos/3_Evaluate_SAE.ipynb) (includes metrics ike L0, cosine similarity, and reconstruction loss)

To load an SAE (see notebook for details):
```
from huggingface_hub import hf_hub_download, list_repo_files
from vit_prisma.sae import SparseAutoencoder

# Step 1: Download SAE weights and config from Hugginface
repo_id = "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-10-hook_mlp_out-l1-1e-05" # Change this to your chosen SAE. See docs/sae_table.md for a full list of SAEs.
sae_path = hf_hub_download(repo_id, file_name="weights.pt") # file_name is usually weights.pt but may have slight naming variation. See the original HF repo for the exact file name
hf_hub_download(repo_id, config_name="config.json")

# Step 2: Load the pretrained SAE weights from the downloaded path
sae = SparseAutoencoder.load_from_pretrained(sae_path) # This now automatically gets config.json and converts into the VisionSAERunnerConfig object
```


## Suite of Pretrained Vision SAE Weights

**For a full list of SAEs for all layers, including CLIP top k, CLIP transcoders, and DINO SAEs, see [here](https://github.com/Prisma-Multimodal/ViT-Prisma/blob/main/docs/sae_table.md).**

We recommend starting with the vanilla CLIP SAEs, which are the highest quality. If you are just getting started with steering CLIP's output, we recommend using the [Layer 11 resid-post SAE](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-1e-05).

### CLIP Vanilla SAEs (All Patches)

| Model | Layer | Sublayer   | l1 coeff. | % Explained var. | Avg L0  | Avg CLS L0 | Cos sim | Recon cos sim | CE    | Recon CE | Zero abl CE | % CE recovered | % Alive features |
|--------|-------|------------|-----------|------------------|---------|-------------|---------|----------------|--------|-----------|--------------|----------------|------------------|
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-0-hook_mlp_out-l1-1e-05) | 0     | mlp_out    | 1e-5      | 98.7             | 604.44  | 36.92       | 0.994   | 0.998          | 6.762  | 6.762     | 6.779        | 99.51          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-0-hook_resid_post-l1-1e-05) | 0     | resid_post | 1e-5      | 98.6             | 1110.9  | 40.46       | 0.993   | 0.988          | 6.762  | 6.763     | 6.908        | 99.23          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-1-hook_mlp_out-l1-1e-05) | 1     | mlp_out    | 1e-5      | 98.4             | 1476.8  | 97.82       | 0.992   | 0.994          | 6.762  | 6.762     | 6.889        | 99.40          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-1-hook_resid_post-l1-1e-05) | 1     | resid_post | 1e-5      | 98.3             | 1508.4  | 27.39       | 0.991   | 0.989          | 6.762  | 6.763     | 6.908        | 99.02          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-2-hook_mlp_out-l1-1e-05) | 2     | mlp_out    | 1e-5      | 98.0             | 1799.7  | 376.0       | 0.992   | 0.998          | 6.762  | 6.762     | 6.803        | 99.44          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-2-hook_resid_post-l1-5e-05) | 2     | resid_post | 5e-5      | 90.6             | 717.84  | 10.11       | 0.944   | 0.960          | 6.762  | 6.767     | 6.908        | 96.34          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-3-hook_mlp_out-l1-1e-05) | 3     | mlp_out    | 1e-5      | 98.1             | 1893.4  | 648.2       | 0.992   | 0.999          | 6.762  | 6.762     | 6.784        | 99.54          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-3-hook_resid_post-l1-1e-05) | 3     | resid_post | 1e-5      | 98.1             | 2053.9  | 77.90       | 0.989   | 0.996          | 6.762  | 6.762     | 6.908        | 99.79          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-4-hook_mlp_out-l1-1e-05) | 4     | mlp_out    | 1e-5      | 98.1             | 1901.2  | 1115.0      | 0.993   | 0.999          | 6.762  | 6.762     | 6.786        | 99.55          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-4-hook_resid_post-l1-1e-05) | 4     | resid_post | 1e-5      | 98.0             | 2068.3  | 156.7       | 0.989   | 0.997          | 6.762  | 6.762     | 6.908        | 99.74          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-5-hook_mlp_out-l1-1e-05) | 5     | mlp_out    | 1e-5      | 98.2             | 1761.5  | 1259.0      | 0.993   | 0.999          | 6.762  | 6.762     | 6.797        | 99.76          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-5-hook_resid_post-l1-1e-05) | 5     | resid_post | 1e-5      | 98.1             | 1953.8  | 228.5       | 0.990   | 0.997          | 6.762  | 6.762     | 6.908        | 99.80          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-6-hook_mlp_out-l1-1e-05) | 6     | mlp_out    | 1e-5      | 98.3             | 1598.0  | 1337.0      | 0.993   | 0.999          | 6.762  | 6.762     | 6.789        | 99.83          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-6-hook_resid_post-l1-1e-05) | 6     | resid_post | 1e-5      | 98.2             | 1717.5  | 321.3       | 0.991   | 0.996          | 6.762  | 6.762     | 6.908        | 99.93          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-7-hook_mlp_out-l1-1e-05) | 7     | mlp_out    | 1e-5      | 98.2             | 1535.3  | 1300.0      | 0.992   | 0.999          | 6.762  | 6.762     | 6.796        | 100.17         | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-7-hook_resid_post-l1-1e-05) | 7     | resid_post | 1e-5      | 98.2             | 1688.4  | 494.3       | 0.991   | 0.995          | 6.762  | 6.761     | 6.908        | 100.24         | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-8-hook_mlp_out-l1-1e-05) | 8     | mlp_out    | 1e-5      | 97.8             | 1074.5  | 1167.0      | 0.990   | 0.998          | 6.762  | 6.761     | 6.793        | 100.57         | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-8-hook_resid_post-l1-1e-05) | 8     | resid_post | 1e-5      | 98.2             | 1570.8  | 791.3       | 0.991   | 0.992          | 6.762  | 6.761     | 6.908        | 100.41         | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-9-hook_mlp_out-l1-1e-05) | 9     | mlp_out    | 1e-5      | 97.6             | 856.68  | 1076.0      | 0.989   | 0.998          | 6.762  | 6.762     | 6.792        | 100.28         | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-9-hook_resid_post-l1-1e-05) | 9     | resid_post | 1e-5      | 98.2             | 1533.5  | 1053.0      | 0.991   | 0.989          | 6.762  | 6.761     | 6.908        | 100.32         | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-10-hook_mlp_out-l1-1e-05) | 10    | mlp_out    | 1e-5      | 98.1             | 788.49  | 965.5       | 0.991   | 0.998          | 6.762  | 6.762     | 6.772        | 101.50         | 99.80            |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-10-hook_resid_post-l1-1e-05) | 10    | resid_post | 1e-5      | 98.4             | 1292.6  | 1010.0      | 0.992   | 0.987          | 6.762  | 6.760     | 6.908        | 100.83         | 99.99            |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_mlp_out-l1-5e-05) | 11    | mlp_out    | 5e-5      | 89.7             | 748.14  | 745.5       | 0.972   | 0.993          | 6.762  | 6.759     | 6.768        | 135.77         | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-1e-05) | 11    | resid_post | 1e-5      | 98.4             | 1405.0  | 1189.0      | 0.993   | 0.987          | 6.762  | 6.765     | 6.908        | 98.03          | 99.99            |

## DINO (Vanilla, all patches)

| Model | Layer | Sublayer | Avg L0 | % Explained var. | Avg CLS L0 | Cos sim | CE | Recon CE | Zero abl CE | % CE Recovered |
|-------|-------|----------|--------|------------------|-------------|----------|------|-----------|---------------|----------------|
| [link](https://huggingface.co/Prisma-Multimodal/DINO-vanilla-x64-all_patches_0-resid_post-507-98)  | 0  | resid_post | 507  | 98 | 347  | 0.95009 | 1.885033 | 1.936518 | 7.2714 | 99.04 |
| [link](https://huggingface.co/Prisma-Multimodal/DINO-vanilla-x64-all_patches_1-resid_post-549-95)  | 1  | resid_post | 549  | 95 | 959  | 0.93071 | 1.885100 | 1.998274 | 7.2154 | 97.88 |
| [link](https://huggingface.co/Prisma-Multimodal/DINO-vanilla-x64-all_patches_2-resid_post-661-95)  | 2  | resid_post | 812  | 95 | 696  | 0.95600 | 1.885134 | 2.006115 | 7.2015 | 97.72 |
| [link](https://huggingface.co/Prisma-Multimodal/DINO-vanilla-x64-all_patches_3-resid_post-989-95)  | 3  | resid_post | 989  | 95 | 616  | 0.96315 | 1.885131 | 1.961913 | 7.2068 | 98.56 |
| [link](https://huggingface.co/Prisma-Multimodal/DINO-vanilla-x64-all_patches_4-resid_post-876-99)  | 4  | resid_post | 876  | 99 | 845  | 0.99856 | 1.885224 | 1.883169 | 7.1636 | 100.04 |
| [link](https://huggingface.co/Prisma-Multimodal/DINO-vanilla-x64-all_patches_5-resid_post-1001-98) | 5  | resid_post | 1001 | 98 | 889  | 0.99129 | 1.885353 | 1.875520 | 7.1412 | 100.19 |
| [link](https://huggingface.co/Prisma-Multimodal/DINO-vanilla-x64-all_patches_6-resid_post-962-99)  | 6  | resid_post | 962  | 99 | 950  | 0.99945 | 1.885239 | 1.872594 | 7.1480 | 100.24 |
| [link](https://huggingface.co/Prisma-Multimodal/DINO-vanilla-x64-all_patches_7-resid_post-1086-98) | 7  | resid_post | 1086 | 98 | 1041 | 0.99341 | 1.885371 | 1.869443 | 7.1694 | 100.30 |
| [link](https://huggingface.co/Prisma-Multimodal/DINO-vanilla-x64-all_patches_8-resid_post-530-90)  | 8  | resid_post | 530  | 90 | 529  | 0.94750 | 1.885511 | 1.978638 | 7.1315 | 98.22 |
| [link](https://huggingface.co/Prisma-Multimodal/DINO-vanilla-x64-all_patches_9-resid_post-1105-99) | 9  | resid_post | 1105 | 99 | 1090 | 0.99541 | 1.885341 | 1.894026 | 7.0781 | 99.83 |
| [link](https://huggingface.co/Prisma-Multimodal/DINO-vanilla-x64-all_patches_10-resid_post-835-99) | 10 | resid_post | 835  | 99 | 839  | 0.99987 | 1.885371 | 1.884487 | 7.3606 | 100.02 |
| [link](https://huggingface.co/Prisma-Multimodal/DINO-vanilla-x64-all_patches_11-resid_post-1085-99) | 11 | resid_post | 1085 | 99 | 1084 | 0.99673 | 1.885370 | 1.911608 | 6.9078 | 99.48 |

## CLIP Transcoders
*CLIP Top-K transcoder performance metrics for all patches.*

| Model                                                                 | Layer | Block | % Explained var. | k    | Avg CLS L0 | Cos sim | CE     | Recon CE | Zero abl CE | % CE recovered |
|-----------------------------------------------------------------------|-------|-------|------------------|------|------------|---------|--------|----------|-------------|----------------|
| [link](https://huggingface.co/Prisma-Multimodal/CLIP-transcoder-topk-768-x64-all_patches_0-mlp-96)  | 0     | MLP   | 96               | 768  | 767        | 0.9655  | 6.7621 | 6.7684   | 6.8804      | 94.68          |
| [link](https://huggingface.co/Prisma-Multimodal/CLIP-transcoder-topk-256-x64-all_patches_1-mlp-94)  | 1     | MLP   | 94               | 256  | 255        | 0.9406  | 6.7621 | 6.7767   | 6.8816      | 87.78          |
| [link](https://huggingface.co/Prisma-Multimodal/CLIP-transcoder-topk-1024-x64-all_patches_2-mlp-93) | 2     | MLP   | 93               | 1024 | 475        | 0.9758  | 6.7621 | 6.7681   | 6.7993      | 83.92          |
| [link](https://huggingface.co/Prisma-Multimodal/CLIP-transcoder-topk-1024-x64-all_patches_3-mlp-90) | 3     | MLP   | 90               | 1024 | 825        | 0.9805  | 6.7621 | 6.7642   | 6.7999      | 94.42          |
| [link](https://huggingface.co/Prisma-Multimodal/CLIP-transcoder-topk-512-x64-all_patches_4-mlp-76)  | 4     | MLP   | 76               | 512  | 29         | 0.9830  | 6.7621 | 6.7636   | 6.8080      | 96.76          |
| [link](https://huggingface.co/Prisma-Multimodal/CLIP-transcoder-topk-1024-x64-all_patches_5-mlp-91) | 5     | MLP   | 91               | 1024 | 1017       | 0.9784  | 6.7621 | 6.7643   | 6.8296      | 96.82          |
| [link](https://huggingface.co/Prisma-Multimodal/CLIP-transcoder-topk-1024-x64-all_patches_6-mlp-94) | 6     | MLP   | 94               | 1024 | 924        | 0.9756  | 6.7621 | 6.7630   | 6.8201      | 98.40          |
| [link](https://huggingface.co/Prisma-Multimodal/CLIP-transcoder-topk-1024-x64-all_patches_7-mlp-97) | 7     | MLP   | 97               | 1024 | 1010       | 0.9629  | 6.7621 | 6.7631   | 6.8056      | 97.68          |
| [link](https://huggingface.co/Prisma-Multimodal/CLIP-transcoder-topk-1024-x64-all_patches_8-mlp-98) | 8     | MLP   | 98               | 1024 | 1023       | 0.9460  | 6.7621 | 6.7630   | 6.8017      | 97.70          |
| [link](https://huggingface.co/Prisma-Multimodal/CLIP-transcoder-topk-1024-x64-all_patches_9-mlp-98) | 9     | MLP   | 98               | 1024 | 1023       | 0.9221  | 6.7621 | 6.7630   | 6.7875      | 96.50          |
| [link](https://huggingface.co/Prisma-Multimodal/CLIP-transcoder-topk-1024-x64-all_patches_10-mlp-97)| 10    | MLP   | 97               | 1024 | 1019       | 0.9334  | 6.7621 | 6.7636   | 6.7860      | 93.95          |


More details are in our whitepaper [here](https://arxiv.org/abs/2504.19475). For more SAEs, including CLS-only and spatial patch-only variants, see the [SAE table](/docs/sae_table.md). We've also visualized some Prisma SAEs [here](https://semanticlens.hhi-research-insights.de/umap-view).

# Basic Mechanistic Interpretability
An earlier version of Prisma included features for basic mechanistic interpretability, including the logit lens and attention head visualizations. In addition to the tutorial notebooks below, you can also check out this [corresponding talk](https://youtu.be/gQbh-RZtsq4?t=0) on some of these techniques.

1. [Main ViT Demo](https://colab.research.google.com/drive/1TL_BY1huQ4-OTORKbiIg7XfTyUbmyToQ) - Overview of main mechanistic interpretability technique on a ViT, including direct logit attribution, attention head visualization, and activation patching. The activation patching switches the net's prediction from tabby cat to Border collie with a minimum ablation.
2. [Emoji Logit Lens](https://colab.research.google.com/drive/1yAHrEoIgkaVqdWC4GY-GQ46ZCnorkIVo) - Deeper dive into layer- and patch-level predictions with interactive plots.
3. [Interactive Attention Head Tour](https://colab.research.google.com/drive/1P252fCvTHNL_yhqJDeDVOXKCzIgIuAz2) - Deeper dive into the various types of attention heads a ViT contains with interactive JavaScript.

## Features
For a demo of Prisma's mech interp features, including the visualizations below with interactivity, check out the demo notebooks above.

**Attention head visualization**

<img src="https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/assets/images/corner-head.gif" width="300">

<div style="display: flex; align-items: center;">
  <img src="assets/images/attention head 1.png" alt="Logo Image 1" width="250" style="margin-right: 10px;"/>
  <img src="assets/images/attention head 2.png" alt="Logo Image 2" width="250" style="margin-right: 10px;"/>
  <img src="assets/images/attention head 3.png" alt="Logo Image 3" width="250"/>
</div>

**Activation patching**

<img src="assets/images/patched head.png" width="400">

**Direct logit attribution**

<img src="assets/images/direct logit attribution.png" width="600">

**Emoji logit lens**

<div style="display: flex; align-items: center;">
<img src="assets/images/dogit lens 2.png" width="400">
<img src="assets/images/cat toilet segmentation.png" width="400">
<img src="assets/images/child lion segmentation.png" width="400">
<img src="assets/images/cheetah segmentation.png" width="400">

</div>

## Custom Models & Checkpoints

### ImageNet-1k classification checkpoints (patch size 32)

All models include training checkpoints, in case you want to analyze training dynamics.

This larger patch size ViT has inspectable attention heads; else the patch size 16 attention heads are too large to easily render in JavaScript.


| **Size** | **NumLayers** | **Attention+MLP** | **AttentionOnly** | **Model Link**                              |
|:--------:|:-------------:|:-----------------:|:-----------------:|--------------------------------------------|
| **tiny** | **3**         | 0.22 \| 0.42 |            N/A            | [Attention+MLP](https://huggingface.co/PraneetNeuro/ImageNet-Small-Attention-and-MLP-Patch32) |

### ImageNet-1k classification checkpoints (patch size 16)

The detailed training logs and metrics can be found [here](https://wandb.ai/vit-prisma/Imagenet/overview?workspace=user-yash-vadi). These models were trained by Yash Vadi.

**Table of Results**

Accuracy `[ <Acc> | <Top5 Acc> ]`

| **Size** | **NumLayers** | **Attention+MLP** | **AttentionOnly** | **Model Link**                              |
|:--------:|:-------------:|:-----------------:|:-----------------:|--------------------------------------------|
| **tiny** | **1**         | 0.16 \| 0.33             |  0.11 \| 0.25             | [AttentionOnly](https://huggingface.co/IamYash/ImageNet-tiny-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/ImageNet-tiny-Attention-and-MLP) |
| **base** | **2**         | 0.23 \| 0.44             |  0.16 \| 0.34             | [AttentionOnly](https://huggingface.co/IamYash/ImageNet-base-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/ImageNet-base-Attention-and-MLP) |
| **small**| **3**         | 0.28 \| 0.51            | 0.17 \| 0.35             | [AttentionOnly](https://huggingface.co/IamYash/ImageNet-small-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/ImageNet-small-Attention-and-MLP) |
| **medium**|**4**         | 0.33 \| 0.56             | 0.17 \| 0.36             | [AttentionOnly](https://huggingface.co/IamYash/ImageNet-medium-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/ImageNet-medium-Attention-and-MLP) |


# Contributors
This library was originally founded by [Sonia Joseph](https://github.com/soniajoseph), alongside fantastic contributors: [Praneet Suresh](https://github.com/PraneetNeuro), [Yash Vadi](https://github.com/YashVadi), [Rob Graham](https://github.com/themachinefan), [Lorenz Hufe](https://github.com/lowlorenz), [Edward Stevinson](https://github.com/stevinson), and [Ethan Goldfarb](https://github.com/ekg15), _and more coming soon_. You learn more about our contributions on our [Contributors page](https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/PRISMA_CONTRIBUTORS.md) (coming soon). Also, check out [our whitepaper](https://arxiv.org/abs/2504.19475).

Thank you to Leo Gao, Joseph Bloom, Lee Sharkey, Neel Nanda, and Yossi Gandelsman for the feedback and discussions at the beginning of this repo's development.

We welcome new contributors. Check out our contributing guidelines [here](CONTRIBUTING.md) and our [open Issues](https://github.com/soniajoseph/ViT-Prisma/issues).

# Citation

Please cite this repository when used in papers or research projects. Thank you for supporting the community! 
```
@misc{joseph2025prismaopensourcetoolkit,
      title={Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video}, 
      author={Sonia Joseph and Praneet Suresh and Lorenz Hufe and Edward Stevinson and Robert Graham and Yash Vadi and Danilo Bzdok and Sebastian Lapuschkin and Lee Sharkey and Blake Aaron Richards},
      year={2025},
      eprint={2504.19475},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.19475}, 
}
```
```
@misc{joseph2023vit,
  author = {Sonia Joseph},
  title = {ViT Prisma: A Mechanistic Interpretability Library for Vision Transformers},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/soniajoseph/vit-prisma}}
}
```

# License

We have an MIT license [here](https://github.com/soniajoseph/ViT-Prisma/blob/main/LICENSE).
