'''
Build zero shot classifier for CLIP to classify image embeddings into your choice of text classes in the CLIP semantic space.
'''

from vit_prisma.utils.openai_templates import OPENAI_IMAGENET_TEMPLATES
from vit_prisma.dataloaders.imagenet_classes_simple import imagenet_classes
from vit_prisma.utils.data_utils.imagenet.imagenet100_classes import imagenet100_classes

import torch
import tqdm


import argparse


import open_clip

import numpy as np

import torch.nn.functional as F

import os


# import Path
from pathlib import Path

def get_args_parser():
    parser = argparse.ArgumentParser('Get classifier weights', add_help=False)
    # Model parameters
    parser.add_argument('--model_name', default='hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K', type=str, metavar='MODEL',
                        help='Name of model to use')
    parser.add_argument('--dataset', default='imagenet', help='waterbirds, imagenet, or imagenet100')
    # Dataset parameters
    parser.add_argument('--output_dir', default='/network/scratch/s/firstname.joseph/clip_benchmark',
                        help='path where to save')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for testing')
    return parser


@torch.no_grad()
@torch.cuda.amp.autocast()
def zero_shot_classifier(model, tokenizer, classnames, templates, 
                         device, amp=True, use_format=False):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.
    

    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    classnames: list of str
        name of classes
    
    templates: list of str
        templates to use.
    
    Returns
    -------
    
    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    zeroshot_weights = []
    for classname in tqdm.tqdm(classnames, desc="Building zero-shot classifier"):
        texts = [template.format(c=classname) if use_format else template(classname) for template in templates]
        texts = tokenizer(texts).to(device)  # tokenize
        class_embeddings = model.encode_text(texts)
        class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def clean_model_name(model_name):
    # Replace hyphens, dashes, slashes, and colons with underscores
    replacements = [
        ('-', '_'),  # hyphen
        ('—', '_'),  # em dash
        ('–', '_'),  # en dash
        ('/', '_'),  # slash
        (':', '_'),  # colon
    ]
    
    cleaned_name = model_name
    for old, new in replacements:
        cleaned_name = cleaned_name.replace(old, new)
    
    return cleaned_name


def build_zero_shot_classifier(args):
    """Calculates the classifier projection weights."""
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.to(args.device)
    model.eval()
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    
    classes = {
        'imagenet': imagenet_classes, 
        'imagenet100': [v.split(',')[0].strip() for v in imagenet100_classes.values()]
,
        }[args.dataset]
    classifier = zero_shot_classifier(model, tokenizer, classes, OPENAI_IMAGENET_TEMPLATES, args.device)

    for i, classname in enumerate(list(classes)[:10]):
        print(f"{i}: {classname}")

    clean_name = clean_model_name(args.model_name)

    os.makedirs(args.output_dir, exist_ok=True )

    with open(os.path.join(args.output_dir, f'{args.dataset}_classifier_{clean_name}.npy'), 'wb') as f:
        np.save(f, classifier.detach().cpu().numpy())
        print(f"Saved classifier weights to {args.output_dir}/{args.dataset}_classifier_{clean_name}.npy")

    return classifier

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    build_zero_shot_classifier(args)
