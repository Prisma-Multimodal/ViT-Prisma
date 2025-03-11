# ImageNet-1k Results
In our ImageNet experiment, we aimed to assess the performance of Mice ViTs on a more complex and diverse dataset, ImageNet.  We trained mice ViTs on classifying the 1000 ImageNet classes.

## Training Details
Siinstitutionr to the dSprites experiment, for each attention layer setting, we explored two model variants: an attention-only model and a model combining attention with the MLP module. Dropout and layer normalization were not applied for simplicity. The detailed training logs and metrics can be found [here](https://wandb.ai/vit-prisma/Imagenet/overview?workspace=user-yash-vadi).

## Table of Results
Below table describe the accuracy `[ <Acc> | <Top5 Acc> ]` of Mice ViTs with different configurations.
| **Size** | **NumLayers** | **Attention+MLP** | **AttentionOnly** | **Model Link**                              |
|:--------:|:-------------:|:-----------------:|:-----------------:|--------------------------------------------|
| **tiny** | **1**         | 0.16 \| 0.33             |  0.11 \| 0.25             | [AttentionOnly](https://huggingface.co/IamYash/ImageNet-tiny-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/ImageNet-tiny-Attention-and-MLP) |
| **base** | **2**         | 0.23 \| 0.44             |  0.16 \| 0.34             | [AttentionOnly](https://huggingface.co/IamYash/ImageNet-base-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/ImageNet-base-Attention-and-MLP) |
| **small**| **3**         | 0.28 \| 0.51            | 0.17 \| 0.35             | [AttentionOnly](https://huggingface.co/IamYash/ImageNet-small-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/ImageNet-small-Attention-and-MLP) |
| **medium**|**4**         | 0.33 \| 0.56             | 0.17 \| 0.36             | [AttentionOnly](https://huggingface.co/IamYash/ImageNet-medium-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/ImageNet-medium-Attention-and-MLP) |
