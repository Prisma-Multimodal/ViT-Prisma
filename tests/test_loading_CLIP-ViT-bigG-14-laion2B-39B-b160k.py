import pytest
import torch
import timm
from vit_prisma.models.base_vit import HookedViT
import open_clip

import numpy as np

import os

from vit_prisma.model_eval.evaluate_imagenet import zero_shot_eval
from vit_prisma.dataloaders.imagenet_dataset import load_imagenet

from transformers import CLIPVisionModelWithProjection

import torch

# For OpenCLIP models
cache_dir = '/network/scratch/s/firstname.joseph/hub'
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['OPEN_CLIP_CACHE'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

og_model_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"


@pytest.mark.skip(reason="Running out of space on Github Actions device")
def test_output():
    random_input = torch.rand((1, 3, 224, 224))
    hooked_vit_g = HookedViT.from_pretrained(og_model_name, is_clip=True, is_timm=False, fold_ln=False)
    hooked_vit_op = hooked_vit_g(random_input)


    hf_model = CLIPVisionModelWithProjection.from_pretrained(og_model_name, cache_dir=cache_dir)
    hf_model.eval()
    op = hf_model(random_input)

    print("Do does output match?")
    print(torch.allclose(hooked_vit_op, op.image_embeds, atol=1e-4))

@pytest.mark.skip(reason="TODO: Reliant on files not in repo")
def test_accuracy_baseline_og_model():
    parent_dir = '/network/scratch/s/firstname.joseph/clip_benchmark/'
    classifier = np.load(os.path.join(parent_dir, 'imagenet_classifier_hf_hub_laion_CLIP_ViT_bigG_14_laion2B_39B_b160k.npy'))

    model_name = 'hf-hub:' + og_model_nam
    og_model, _, preprocess = open_clip.create_model_and_transforms(model_name, cache_dir)
    og_model.eval()

    print("Classifier and model loaded")

    data = {}
    dataset_path =  "/network/scratch/s/firstname.joseph/datasets/kaggle_datasets"
    data['imagenet-val'] = load_imagenet(preprocess_transform=preprocess, dataset_path=dataset_path, dataset_type='imagenet1k-val')
    epoch = 1
    results = zero_shot_eval(og_model, data, epoch, model_name=model_name, pretrained_classifier=classifier)
    print("Results", results)
    # I get 0.6918 on ImageNet Val; benchmarked in ML Foundations OpenCLIP repo is 0.6917

@pytest.mark.skip(reason="TODO: Reliant on files not in repo")
def test_accuracy_baseline_hooked_model():
    parent_dir = '/network/scratch/s/firstname.joseph/clip_benchmark/'
    classifier = np.load(os.path.join(parent_dir, 'imagenet_classifier_hf_hub_laion_CLIP_ViT_bigG_14_laion2B_39B_b160k.npy'))

    hooked_model = HookedViT.from_pretrained(og_model_name, is_timm=False, is_clip=True, fold_ln=False, center_writing_weights=False) # in future, do all models
    hooked_model.to('cuda')
    hooked_model.eval()

    print("Classifier and model loaded")

    model_name = 'hf-hub:' + og_model_name
    og_model, _, preprocess = open_clip.create_model_and_transforms(model_name) # just need preprocessor
    del og_model

    print(preprocess)

    data = {}
    dataset_path =  "/network/scratch/s/firstname.joseph/datasets/kaggle_datasets"
    data['imagenet-val'] = load_imagenet(preprocess_transform=preprocess, dataset_path=dataset_path, dataset_type='imagenet1k-val')
    epoch = 1
    results = zero_shot_eval(hooked_model, data, epoch, model_name=og_model_name, pretrained_classifier=classifier)
    print("Results", results)
