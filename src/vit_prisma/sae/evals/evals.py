import torch
import torchvision

import plotly.express as px

from tqdm import tqdm

import einops
from typing import List

import argparse

import numpy as np
import os
import requests


from plotly.io import write_image
import textwrap

from dataclasses import dataclass
from vit_prisma.sae.config import VisionModelSAERunnerConfig

# from vit_prisma.sae.train_sae import VisionSAETrainer
from torch.utils.data import DataLoader


from vit_prisma.utils.data_utils.imagenet.imagenet_utils import setup_imagenet_paths
from vit_prisma.dataloaders.imagenet_dataset import ImageNetValidationDataset
from vit_prisma.models.base_vit import HookedViT

from vit_prisma.sae.sae import SparseAutoencoder

from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_index_to_name


import matplotlib.pyplot as plt

from typing import Any, List, Tuple, Dict

import json


# import cross-entropy loss
import torch.nn.functional as F
# import partial
from functools import partial


class IndexedDataset(torch.utils.data.Dataset):
    """Wrapper dataset that adds indices to the data."""
    
    def __init__(self, dataset):
        """
        Args:
            dataset: The dataset to wrap
        """
        self.dataset = dataset
    
    def __getitem__(self, idx):
        """
        Get item with index.
        
        Args:
            idx: The index
            
        Returns:
            data, label, idx
        """
        # Get the original data
        data = self.dataset[idx]
        
        # If the dataset returns a tuple, assume first item is data, second is label
        if isinstance(data, tuple) and len(data) == 2:
            return data[0], data[1], idx
        # If it returns something else, just add the index
        else:
            return data, idx
    
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataset)
    

@dataclass
class EvalConfig(VisionModelSAERunnerConfig):
    sae_path: str = '/network/scratch/s/sonia.joseph/sae_checkpoints/tinyclip_40M_mlp_out/1f89d99e-wkcn-TinyCLIP-ViT-40M-32-Text-19M-LAION400M-expansion-16/n_images_520028.pt'
    model_name: str = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"
    model_type: str =  "clip"
    patch_size: str = 32

    dataset_path: str = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets"
    dataset_train_path: str = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/train"
    dataset_val_path: str = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/val"

    verbose: bool = True

    device: bool = 'cuda'

    eval_max: int = 50_000 # Number of images to evaluate
    batch_size: int = 64

    samples_per_bin: int = 10 # Number of features to sample per pre-specified interval
    max_images_per_feature: int = 20 # Number of max images to collect per feature
    sampling_type: str = 'avg'


    @property
    def max_image_output_folder(self) -> str:

        # Create a new folder path in sae_checkpoints/images with the original name
        parent_dir = os.path.dirname(self.sae_path)

        output_folder = os.path.join(parent_dir, 'max_images')

        # Ensure the directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        return output_folder
    
    @property
    def save_figure_dir(self) -> str:

        parent_dir = os.path.dirname(self.sae_path)

        # Create a new folder path in sae_checkpoints/images with the original name
        output_folder = os.path.join(parent_dir, 'save_fig_dir')

        # Ensure the directory exists
        os.makedirs(output_folder, exist_ok=True)
        return output_folder   

def create_eval_config(args):
    return EvalConfig(
        sae_path=args.sae_path,
        model_name=args.model_name,
        model_type=args.model_type,
        patch_size=args.patch_size,
        dataset_path=args.dataset_path,
        dataset_train_path=args.dataset_train_path,
        dataset_val_path=args.dataset_val_path,
        device=args.device,
        verbose=args.verbose,
        eval_max=args.eval_max,
        batch_size=args.batch_size,
        samples_per_bin=args.samples_per_bin,
        sampling_type=args.sampling_type,

    )

def setup_environment():
    torch.set_grad_enabled(False)

def load_model(cfg):
    from vit_prisma.models.base_vit import HookedViT
    model = HookedViT.from_pretrained(cfg.model_name, is_timm=False, is_clip=True).to(cfg.device)
    model.eval()
    return model

# def load_sae(cfg):
#     sparse_autoencoder = SparseAutoencoder(cfg).load_from_pretrained(cfg.sae_path)
#     sparse_autoencoder.to(cfg.device)
#     sparse_autoencoder.eval()  # prevents error if we're expecting a dead neuron mask for who 
#     return sparse_autoencoder

def load_sae(cfg):
    sae_path = cfg.sae_path
    # config_path = os.path.dirname(sae_path) + '/config.json'
    # config = VisionModelSAERunnerConfig.load_config(config_path)
    sae = SparseAutoencoder.load_from_pretrained(sae_path)
    return sae

def save_stats(sae_path, stats):
    # Unpack the stats tuple
    avg_loss, avg_cos_sim, avg_reconstruction_loss, avg_zero_abl_loss, avg_l0, avg_l0_cls, avg_l0_img, log_frequencies_per_token, log_frequencies_per_image = stats

    # Get the parent directory and filename without extension
    parent_dir = os.path.dirname(sae_path)
    filename_without_ext = os.path.splitext(os.path.basename(sae_path))[0]
    
    # Create the stats filename
    stats_filename = f"{filename_without_ext}_stats.json"
    stats_path = os.path.join(parent_dir, stats_filename)
    
    # Prepare the stats dictionary
    stats_dict = {
        "avg_loss": float(avg_loss),
        "avg_cos_sim": float(avg_cos_sim),
        "avg_reconstruction_loss": float(avg_reconstruction_loss),
        "avg_zero_abl_loss": float(avg_zero_abl_loss),
        "avg_l0": float(avg_l0),
        "avg_l0_cls": float(avg_l0_cls),
        "avg_l0_img": float(avg_l0_img),
        "log_frequencies_per_token": log_frequencies_per_token.tolist() if isinstance(log_frequencies_per_token, np.ndarray) else log_frequencies_per_token,
        "log_frequencies_per_image": log_frequencies_per_image.tolist() if isinstance(log_frequencies_per_image, np.ndarray) else log_frequencies_per_image
    }
    
    # Custom JSON encoder to handle NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super(NumpyEncoder, self).default(obj)

    # Save the stats to a JSON file
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=4, cls=NumpyEncoder)
    
    print(f"Stats saved to {stats_path}")


def get_imagenet_val_dataset_visualize(cfg):
    imagenet_paths = setup_imagenet_paths(cfg.dataset_path)

    return ImageNetValidationDataset(imagenet_paths['val'], 
                                imagenet_paths['label_strings'], 
                                imagenet_paths['val_labels'],
                                torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),]), return_index=True)

def average_l0_test(cfg, val_dataloader, sparse_autoencoder, model, evaluation_max=100):
    total_l0 = []
    with torch.no_grad():
        for i in range(evaluation_max):
            batch_tokens, labels, indices = next(iter(val_dataloader))
            batch_tokens = batch_tokens.to(cfg.device)
            _, cache = model.run_with_cache(batch_tokens, names_filter = sparse_autoencoder.cfg.hook_point)
            sae_out, feature_acts, loss, mse_loss, l1_loss, _, _ = sparse_autoencoder(
                cache[sparse_autoencoder.cfg.hook_point].to(cfg.device)
            )
            del cache

            # ignore the bos token, get the number of features that activated in each token, averaged accross batch and position
            l0 = (feature_acts[:, :] > 0).float().sum(-1).detach()
            total_l0.append(l0)
    average_l0 = torch.cat(total_l0).mean(0)
    print(f"Average L0: {average_l0.mean()}") if cfg.verbose else None

    # Create histogram using matplotlib
    plt.figure(figsize=(10, 6))
    plt.hist(average_l0.flatten().cpu().numpy(), bins=50, edgecolor='black')
    plt.title("Distribution of Average L0")
    plt.xlabel("Average L0")
    plt.ylabel("Frequency")

    # Save the figure
    save_path = os.path.join(cfg.save_figure_dir, "average_l0.png")
    plt.savefig(save_path)
    plt.close()  # Close the figure to free up memory

    print(f"Saved average l0 figure to {save_path}") if cfg.verbose else None


# due to loading issues with laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90k
def get_text_embeddings_openclip(vanilla_model, processor, tokenizer, original_text, batch_size=32):
    # Split the text into batches
    text_batches = [original_text[i:i+batch_size] for i in range(0, len(original_text), batch_size)]

    all_embeddings = []

    for batch in text_batches:
        inputs = tokenizer(batch)
        # inputs = {k: v.to(cfg.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_embeddings = vanilla_model.encode_text(inputs)

        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        all_embeddings.append(text_embeddings)

    # Concatenate all batches
    final_embeddings = torch.cat(all_embeddings, dim=0)

    return final_embeddings


# # this needs to be redone to not assume huggingface
# def get_text_embeddings(model_name, original_text, batch_size=32):
#     from transformers import CLIPProcessor, CLIPModel
#     # remove clip tag
    
#     model_name = open_clip.create_model_and_transforms('hf-hub:' + model_name.replace('open-clip:', ''))
#     vanilla_model = CLIPModel.from_pretrained(model_name)
#     processor = CLIPProcessor.from_pretrained(model_name, do_rescale=False)

#     # Split the text into batches
#     text_batches = [original_text[i:i+batch_size] for i in range(0, len(original_text), batch_size)]

#     all_embeddings = []

#     for batch in text_batches:
#         inputs = processor(text=batch, return_tensors='pt', padding=True, truncation=True, max_length=77)
#         # inputs = {k: v.to(cfg.device) for k, v in inputs.items()}

#         with torch.no_grad():
#             text_embeddings = vanilla_model.get_text_features(**inputs)

#         text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
#         all_embeddings.append(text_embeddings)

#     # Concatenate all batches
#     final_embeddings = torch.cat(all_embeddings, dim=0)

#     return final_embeddings


def get_text_embeddings(model_name, device):
    model_simple_name = model_name.split("/")[-1]
    text_embedding_path = f'/home/mila/s/sonia.joseph/cvpr-2025/src/clip_circuits/sparse_feature_circuits/greedy_steering/text_data/{model_simple_name}_text_embeddings.npy'
    text_features = np.load(text_embedding_path)
    print(f"Loading precomputed text embeddings from {text_embedding_path}...")
    text_features = torch.tensor(text_features).to(device)
    text_features = F.normalize(text_features)
    return text_features

@torch.no_grad()
def get_substitution_loss(
    sparse_autoencoder: SparseAutoencoder,
    model: HookedViT,
    batch_tokens: torch.Tensor,
    gt_labels: torch.Tensor,
    text_embeddings: torch.Tensor,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    # Move model to device if it's not already there
    model = model.to(device)
    
    # Move all tensors to the same device
    batch_tokens = batch_tokens.to(device)
    gt_labels = gt_labels.to(device)
    text_embeddings = text_embeddings.to(device)

    # Get image embeddings
    image_embeddings, _ = model.run_with_cache(batch_tokens)

    # Calculate similarity scores
    # print(f"image_embeddings.shape: {image_embeddings.shape}")
    # print(f"text_embeddings.shape: {text_embeddings.shape}")

    softmax_values, top_k_indices = get_similarity(image_embeddings, text_embeddings, device=device)
    class_logits = get_logits(image_embeddings, text_embeddings, device=device)
    # print(f"softmax_values: {softmax_values}")
    # print(f"softmax_values.shape: {softmax_values.shape}")
    # print(f"gt_labels.shape: {gt_labels.shape}")
    # print(f"gt_labels: {gt_labels}")
    # print(f"top_k_indices: {top_k_indices}")
    # print(f"class_logits: {class_logits}")
    # Calculate cross-entropy loss
    # cross entropy should take logits, not softmax
    loss_softmax = F.cross_entropy(softmax_values, gt_labels)
    loss = F.cross_entropy(class_logits, gt_labels)
    # Safely extract the loss value
    # print(f"model loss: {loss}")
    # print(f"model loss_logit: {loss_softmax}")
    loss_value = loss.item() if torch.isfinite(loss).all() else float('nan')

    head_index = sparse_autoencoder.cfg.hook_point_head_index
    hook_point = sparse_autoencoder.cfg.hook_point

    def standard_replacement_hook(activations: torch.Tensor, hook: Any):
        activations = sparse_autoencoder.forward(activations)[0].to(activations.dtype)
        return activations

    def head_replacement_hook(activations: torch.Tensor, hook: Any):
        new_activations = sparse_autoencoder.forward(activations[:, :, head_index])[0].to(activations.dtype)
        activations[:, :, head_index] = new_activations
        return activations

    replacement_hook = standard_replacement_hook if head_index is None else head_replacement_hook

    recons_image_embeddings = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(hook_point, partial(replacement_hook))],
    )
    # recons_softmax_values, _ = get_similarity(recons_image_embeddings, text_embeddings, device=device)
    recons_softmax_values = get_logits(recons_image_embeddings, text_embeddings, device=device)
    recons_loss = F.cross_entropy(recons_softmax_values, gt_labels)

    zero_abl_image_embeddings = model.run_with_hooks(
        batch_tokens, fwd_hooks=[(hook_point, zero_ablate_hook)]
    )
    zero_abl_softmax_values = get_logits(zero_abl_image_embeddings, text_embeddings, device=device)
    zero_abl_loss = F.cross_entropy(zero_abl_softmax_values, gt_labels)

    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)

    return score, loss, recons_loss, zero_abl_loss


def get_logits(image_features, text_features, device='cuda'):
    return (image_features.to(device) @ text_features.to(device).T)


def get_similarity(image_features, text_features, k=5, device='cuda'):
  image_features = image_features.to(device)
  text_features = text_features.to(device)

  softmax_values = (image_features @ text_features.T).softmax(dim=-1)
  top_k_values, top_k_indices = torch.topk(softmax_values, k, dim=-1)
  return softmax_values, top_k_indices

def get_text_labels(name='wordbank'):
    """
    Loads the library of logit labels from a GitHub URL.

    Returns:
    list: A list of string labels.
    """
    if name == 'wordbank':
        url = "https://raw.githubusercontent.com/yossigandelsman/clip_text_span/main/text_descriptions/image_descriptions_general.txt"
        try:
            # Fetch the content from the URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Split the content into lines and strip whitespace
            all_labels = [line.strip() for line in response.text.splitlines()]
            
            print(f"Number of labels loaded: {len(all_labels)}")
            print(f"First 5 labels: {all_labels[:5]}")
            return all_labels
        
        except requests.RequestException as e:
            print(f"An error occurred while fetching the labels: {e}")
            return []
    elif name == 'imagenet':
        from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_text_labels
        return get_imagenet_text_labels()
    else:
        raise ValueError(f"Invalid label set name: {name}")
    
def zero_ablate_hook(activations: torch.Tensor, hook: Any):
    activations = torch.zeros_like(activations)
    return activations

def get_feature_probability(feature_acts):
    return (feature_acts.abs() > 0).float().flatten(0, 1)

def calculate_log_frequencies(total_acts, total_tokens):
    # print out all shapes
    feature_probs = total_acts / total_tokens
    log_feature_probs = torch.log10(feature_probs)
    return log_feature_probs.cpu().numpy()

def process_dataset(model, sparse_autoencoder, dataloader, cfg):
    all_l0 = []
    all_l0_cls = []

    # image level l0
    all_l0_image = []

    total_loss = 0
    total_reconstruction_loss = 0
    total_zero_abl_loss = 0
    total_samples = 0
    all_cosine_similarity = []

    model.eval()
    sparse_autoencoder.eval()

    # all_labels = get_text_labels('imagenet')
    # text_embeddings = get_text_embeddings(cfg.model_name, all_labels)

    text_embeddings = get_text_embeddings(cfg.model_name, cfg.device)

    total_acts = None
    total_tokens = 0
    total_images = 0

    with torch.no_grad():
        for batch_tokens, gt_labels, indices in tqdm(dataloader):
            batch_tokens = batch_tokens.to(cfg.device)
            batch_size = batch_tokens.shape[0]

            # batch shape

            total_samples += batch_size

            _, cache = model.run_with_cache(batch_tokens, names_filter=sparse_autoencoder.cfg.hook_point)
            hook_point_activation = cache[sparse_autoencoder.cfg.hook_point].to(cfg.device)
            
            sae_out, feature_acts, loss, mse_loss, l1_loss, _, _ = sparse_autoencoder(hook_point_activation)
    

            # Calculate feature probability
            sae_activations = get_feature_probability(feature_acts)
            if total_acts is None:
                total_acts = sae_activations.sum(0)
            else:
                total_acts += sae_activations.sum(0)
            
            total_tokens += sae_activations.shape[0]
            total_images += batch_size

            # Get L0 stats per token
            l0 = (feature_acts[:, 1:, :] > 0).float().sum(-1).detach()
            all_l0.extend(l0.mean(dim=1).cpu().numpy())
            l0_cls = (feature_acts[:, 0, :] > 0).float().sum(-1).detach()
            all_l0_cls.extend(l0_cls.flatten().cpu().numpy())

            # Get L0 stats per image
            l0 = (feature_acts > 0).float().sum(-1).detach()
            image_l0 = l0.sum(dim=1)  # Sum across all tokens  
            all_l0_image.extend(image_l0.cpu().numpy())

            # Calculate cosine similarity between original activations and sae output
            cos_sim = torch.cosine_similarity(einops.rearrange(hook_point_activation, "batch seq d_mlp -> (batch seq) d_mlp"),
                                                                              einops.rearrange(sae_out, "batch seq d_mlp -> (batch seq) d_mlp"),
                                                                                dim=0).mean(-1).tolist()
            all_cosine_similarity.append(cos_sim)

            # Calculate substitution loss
            score, loss, recons_loss, zero_abl_loss = get_substitution_loss(sparse_autoencoder, model, batch_tokens, gt_labels, 
                                                                      text_embeddings, device=cfg.device)

            total_loss += loss.item()
            total_reconstruction_loss += recons_loss.item()
            total_zero_abl_loss += zero_abl_loss.item()

            if total_samples >= cfg.eval_max:
                break

    # Calculate average metrics
    avg_loss = total_loss / total_samples
    avg_reconstruction_loss = total_reconstruction_loss / total_samples
    avg_zero_abl_loss = total_zero_abl_loss / total_samples
    
    avg_l0 = np.mean(all_l0)
    avg_l0_cls = np.mean(all_l0_cls)
    avg_l0_image = np.mean(all_l0_image)

    avg_cos_sim = np.mean(all_cosine_similarity)
    log_frequencies_per_token = calculate_log_frequencies(total_acts, total_tokens)
    log_frequencies_per_image = calculate_log_frequencies(total_acts, total_images)

    # print out everything above
    print(f"Average L0 (features activated): {avg_l0:.6f}")
    print(f"Average L0 (features activated) per CLS token: {avg_l0_cls:.6f}")
    print(f"Average L0 (features activated) per image: {avg_l0_image:.6f}")
    print(f"Average Cosine Similarity: {avg_cos_sim:.4f}")
    print(f"Average Loss: {avg_loss:.6f}")
    print(f"Average Reconstruction Loss: {avg_reconstruction_loss:.6f}")
    print(f"Average Zero Ablation Loss: {avg_zero_abl_loss:.6f}")

    return avg_loss, avg_cos_sim, avg_reconstruction_loss, avg_zero_abl_loss, avg_l0, avg_l0_cls, avg_l0_image, log_frequencies_per_token, log_frequencies_per_image


def get_intervals_for_sparsities(log_freq):
    # Define intervals and conditions
    intervals = [
        (-8, -6),
        (-6, -5),
        (-5, -4),
        (-4, -3),
        (-3, -2),
        (-2, -1),
        (-float('inf'), -8),
        (-1, float('inf'))
    ]

    conditions = [torch.logical_and(log_freq >= lower, log_freq < upper) for lower, upper in intervals]
    condition_texts = [f"TOTAL_logfreq_[{lower},{upper}]" for lower, upper in intervals]

    # Replace infinity with appropriate text for readability
    condition_texts[-2] = condition_texts[-2].replace('-inf', '-∞')
    condition_texts[-1] = condition_texts[-1].replace('inf', '∞')
    return intervals, conditions, condition_texts

# def highest_activating_tokens(
#     images,
#     model,
#     sparse_autoencoder,
#     W_enc,
#     b_enc,
#     feature_ids: List[int],
# ):
#     '''
#     Returns the indices & values for the highest-activating tokens in the given batch of data.
#     '''
#     with torch.no_grad():
#         # Get the post activations from the clean run
#         _, cache = model.run_with_cache(images)

#     inp = cache[sparse_autoencoder.cfg.hook_point]
#     b, seq_len, _ = inp.shape
#     post_reshaped = einops.rearrange(inp, "batch seq d_mlp -> (batch seq) d_mlp")
    
#     # Compute activations
#     sae_in = post_reshaped - sparse_autoencoder.b_dec
#     acts = einops.einsum(
#         sae_in,
#         W_enc,
#         "... d_in, d_in n -> ... n",
#     )
#     acts = acts + b_enc
#     acts = torch.nn.functional.relu(acts)

#     # Reshape acts to (batch, seq, n_features)
#     acts_reshaped = einops.rearrange(acts, "(batch seq) n_features -> batch seq n_features", batch=b, seq=seq_len)
#     temp_top_indices = {}
#     # # The matrix already only contains features due to the selective W_enc you passed in
#     # for idx, fid in enumerate(feature_ids): # Iterate through every feature id, and store the corresponding top images/tokens
#     #     # Get activations for this feature across all tokens and images
#     #     feature_acts = acts_reshaped[:, :, idx].flatten()
        
#     #     # Get top k activating tokens
#     #     top_acts_values, top_acts_indices = torch.sort(feature_acts, descending=True)

#     #     # Convert flat indices to (image_idx, token_idx) pairs
#     #     image_indices = top_acts_indices // seq_len
#     #     token_indices = top_acts_indices % seq_len

#     #     temp_top_indices[fid] = (list(zip(image_indices.tolist(), token_indices.tolist())), top_acts_values.tolist())

#     temp_top_indices = {}
#     for idx, fid in enumerate(feature_ids):
#         feature_acts = acts_reshaped[:, :, idx].flatten()
#         top_acts_values, top_acts_indices = torch.sort(feature_acts, descending=True)
        
#         image_indices = top_acts_indices // seq_len
#         token_indices = top_acts_indices % seq_len

#         temp_top_indices[fid] = {
#             'image_indices': image_indices.tolist(),
#             'token_indices': token_indices.tolist(),
#             'values': top_acts_values.tolist()
#         }
#     return temp_top_indices

torch.no_grad()
def get_heatmap(
          
          image,
          model,
          sparse_autoencoder,
          feature_id,
          device
): 
    image = image.to(device)
    _, cache = model.run_with_cache(image.unsqueeze(0))

    post_reshaped = einops.rearrange( cache[sparse_autoencoder.cfg.hook_point], "batch seq d_mlp -> (batch seq) d_mlp")
    # Compute activations (not from a fwd pass, but explicitly, by taking only the feature we want)
    # This code is copied from the first part of the 'forward' method of the AutoEncoder class
    sae_in =  post_reshaped - sparse_autoencoder.b_dec # Remove decoder bias as per Anthropic
    acts = einops.einsum(
            sae_in,
            sparse_autoencoder.W_enc[:, feature_id],
            "x d_in, d_in -> x",
        )
    return acts 
     
def image_patch_heatmap(activation_values, cfg):
    patch_size = cfg.patch_size
    image_size = cfg.image_size
    pixel_num = image_size // patch_size
    activation_values = activation_values.detach().cpu().numpy()
    activation_values = activation_values[1:]
    activation_values = activation_values.reshape(pixel_num, pixel_num)

    # Create a heatmap overlay
    heatmap = np.zeros((image_size, image_size))
    patch_size = image_size // pixel_num

    for i in range(pixel_num):
        for j in range(pixel_num):
            heatmap[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = activation_values[i, j]

    return heatmap

# helper functions
update_layout_set = {"xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor", "showlegend", "xaxis_tickmode", "yaxis_tickmode", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap", "coloraxis_showscale"}
def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (torch.Tensor, torch.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")


def wrap_text(text, width=60):
    """Wrap text to a specified width."""
    return "<br>".join(textwrap.wrap(text, width=width))


def hist(cfg, tensor, save_name, show=False, renderer=None, **kwargs):
    '''
    '''
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if "bargap" not in kwargs_post:
        kwargs_post["bargap"] = 0.1
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])

    histogram_fig = px.histogram(x=to_numpy(tensor), **kwargs_pre)

    # Handle the title separately
    title = kwargs_post.pop('title', '')
    wrapped_title = wrap_text(title, width=60)  # Adjust width as needed

    # Update layout
    histogram_fig.update_layout(
        title={
            'text': wrapped_title,
            'font': {'size': 12},
            'y': 0.98,  # Adjust as needed
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(t=120, b=50, l=50, r=50),  # Increased top margin for multi-line title
        height=650,  # Increased height to accommodate wrapped title
        **kwargs_post
    )

    # save_path = os.path.join(cfg.save_figure_dir, f"{save_name}.png")

    # Save the figure as a PNG file
    # Save the figure as PNG and SVG files
    base_path = os.path.join(cfg.save_figure_dir, save_name)
    png_path = f"{base_path}.png"
    svg_path = f"{base_path}.svg"

    write_image(histogram_fig, png_path)
    write_image(histogram_fig, svg_path)
    
    print(f"Histogram saved as PNG: {png_path}")
    print(f"Histogram saved as SVG: {svg_path}")

    if show:
        px.histogram(x=to_numpy(tensor), **kwargs_pre).update_layout(**kwargs_post).show(renderer)
    # close
    


def visualize_sparsities(cfg, log_freq_tokens, log_freq_images, conditions, condition_texts, name, sparse_autoencoder):
        # Visualise sparsities for each instance
        hist(
            cfg,
            log_freq_tokens,
            f"{name}_frequency_tokens_histogram",
            show=False,
            title=f"{name} Log Frequency of Features by Token",
            labels={"x": "log<sub>10</sub>(freq)"},
            histnorm="percent",
            template="ggplot2"
        )
        hist(
            cfg,
            log_freq_images,
            f"{name}_frequency_images_histogram",
            show=False,
            title=f"{name} Log Frequency of Features by Image",
            labels={"x": "log<sub>10</sub>(freq)"},
            histnorm="percent",
            template="ggplot2"
        )

        #TODO these conditions need to be tuned to distribution of your data!

        for condition, condition_text in zip(conditions, condition_texts):
            percentage = (torch.count_nonzero(condition)/log_freq_tokens.shape[0]).item()*100
            if percentage == 0:
                continue
            percentage = int(np.round(percentage))
            rare_encoder_directions = sparse_autoencoder.W_enc[:, condition]
            rare_encoder_directions_normalized = rare_encoder_directions / rare_encoder_directions.norm(dim=0, keepdim=True)

            # Compute their pairwise cosine similarities & sample randomly from this N*N matrix of similarities
            cos_sims_rare = (rare_encoder_directions_normalized.T @ rare_encoder_directions_normalized).flatten()
            cos_sims_rare_random_sample = cos_sims_rare[torch.randint(0, cos_sims_rare.shape[0], (10000,))]

            # Plot results
            hist(
                cfg,
                cos_sims_rare_random_sample,
                f"{name}_low_prop_similarity_{condition_text}",
                show=False,
                marginal="box", 
                title=f"{name} Cosine similarities of random {condition_text} <br> encoder directions with each other ({percentage}% of features)",
                labels={"x": "Cosine sim"},
                histnorm="percent",
                template="ggplot2",
            )



@torch.no_grad()
def compute_feature_activations(
    images: torch.Tensor,
    model: torch.nn.Module,
    sparse_autoencoder: torch.nn.Module,
    encoder_weights: torch.Tensor,
    encoder_biases: torch.Tensor,
    feature_ids: List[int],
    is_cls_list: List[bool],
    top_k: int = 10,
    sampling_type: str = 'avg'
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute the highest activating tokens for given features in a batch of images.
    
    Args:
        images: Input images
        model: The main model
        sparse_autoencoder: The sparse autoencoder
        encoder_weights: Encoder weights for selected features
        encoder_biases: Encoder biases for selected features
        feature_ids: List of feature IDs to analyze
        feature_categories: Categories of the features
        top_k: Number of top activations to return per feature

    Returns:
        Dictionary mapping feature IDs to tuples of (top_indices, top_values)
    """
    _, cache = model.run_with_cache(images, names_filter=[sparse_autoencoder.cfg.hook_point])
    
    layer_activations = cache[sparse_autoencoder.cfg.hook_point]
    batch_size, seq_len, _ = layer_activations.shape

    actual_top_k = min(top_k, batch_size) # Ensure top_k is not greater than batch size
    flattened_activations = einops.rearrange(layer_activations, "batch seq d_mlp -> (batch seq) d_mlp")
    
    sae_input = flattened_activations - sparse_autoencoder.b_dec
    feature_activations = einops.einsum(sae_input, encoder_weights, "... d_in, d_in n -> ... n") + encoder_biases
    feature_activations = torch.nn.functional.relu(feature_activations)
    
    reshaped_activations = einops.rearrange(feature_activations, "(batch seq) d_in -> batch seq d_in", batch=batch_size, seq=seq_len)
    cls_token_activations = reshaped_activations[:, 0, :]
    if sampling_type == 'avg':
        mean_image_activations = reshaped_activations.mean(1)
    else:
        raise ValueError(f"Invalid sampling type: {sampling_type}, or not implemented yet")

    top_activations = {}
    for i, (feature_id, is_cls) in enumerate(zip(feature_ids, is_cls_list)):
        if is_cls:
            top_values, top_indices = cls_token_activations[:, i].topk(actual_top_k)
        else:
            top_values, top_indices = mean_image_activations[:, i].topk(actual_top_k)
        top_activations[feature_id] = (top_indices, top_values)
    
    return top_activations

def find_top_activations(
    val_dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    sparse_autoencoder: torch.nn.Module,
    interesting_features_indices: List[int],
    is_cls_list: List[bool],
    top_k: int = 16,
    max_samples= 50_000,
    batch_size = 54, 
    sampling_type = 'avg'
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Find the top activations for interesting features across the validation dataset.

    Args:
        val_dataloader: Validation data loader
        model: The main model
        sparse_autoencoder: The sparse autoencoder
        interesting_features_indices: Indices of interesting features
        interesting_features_category: Categories of interesting features

    Returns:
        Dictionary mapping feature IDs to tuples of (top_values, top_indices)
    """
    device = next(model.parameters()).device
    top_activations = {i: (None, None) for i in interesting_features_indices}
    encoder_biases = sparse_autoencoder.b_enc[interesting_features_indices]
    encoder_weights = sparse_autoencoder.W_enc[:, interesting_features_indices]

    processed_samples = 0
    for batch_images, _, batch_indices in tqdm(val_dataloader, total=max_samples // batch_size):
        batch_images = batch_images.to(device)
        batch_indices = batch_indices.to(device)
        batch_size = batch_images.shape[0]

        batch_activations = compute_feature_activations(
            batch_images, model, sparse_autoencoder, encoder_weights, encoder_biases,
            interesting_features_indices, is_cls_list, top_k,
            sampling_type=sampling_type,
        )

        for feature_id in interesting_features_indices:
            new_indices, new_values = batch_activations[feature_id]
            new_indices = batch_indices[new_indices]
            
            if top_activations[feature_id][0] is None:
                top_activations[feature_id] = (new_values, new_indices)
            else:
                combined_values = torch.cat((top_activations[feature_id][0], new_values))
                combined_indices = torch.cat((top_activations[feature_id][1], new_indices))
                _, top_k_indices = torch.topk(combined_values, top_k)
                top_activations[feature_id] = (combined_values[top_k_indices], combined_indices[top_k_indices])

        processed_samples += batch_size
        if processed_samples >= max_samples:
            break

    return {i: (values.detach().cpu(), indices.detach().cpu()) 
            for i, (values, indices) in top_activations.items()}


def evaluate(cfg, sparse_autoencoder, model, val_data, val_data_visualize):

    setup_environment()

    print("Loaded SAE config", sparse_autoencoder.cfg) if cfg.verbose else None
    val_data = IndexedDataset(val_data)
    val_dataloader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    print("Loaded model and data") if cfg.verbose else None

    print("Processing dataset...")
    avg_loss, avg_cos_sim, avg_reconstruction_loss, avg_zero_abl_loss, avg_l0, avg_l0_cls, avg_l0_img, log_frequencies_per_token, log_frequencies_per_image = process_dataset(model, sparse_autoencoder, val_dataloader, cfg)

    print("Saving stats...")
    save_stats(cfg.sae_path, (avg_loss, avg_cos_sim, avg_reconstruction_loss, avg_zero_abl_loss, avg_l0, avg_l0_cls, avg_l0_img, log_frequencies_per_token, log_frequencies_per_image))

    print("Plotting log frequencies...")
    log_freq_tokens = torch.Tensor(log_frequencies_per_token)
    log_freq_images = torch.Tensor(log_frequencies_per_image)
    intervals, conditions, conditions_texts = get_intervals_for_sparsities(log_freq_tokens)
    visualize_sparsities(cfg, log_freq_tokens, log_freq_images, conditions, conditions_texts, "TOTAL", sparse_autoencoder)
    # conditions_class = [torch.logical_and(log_freq_class < -4,log_freq_class > -8), log_freq_class <-9, log_freq_class>-4]
    # condition_texts_class = ["logfreq_[-8,-4]", "logfreq_[-inf,-9]","logfreq_[-4,inf]"]
    # visualize_sparsities(log_freq_class, conditions_class, condition_texts_class,"CLS")


    print("Sampling features from pre-specified intervals...")
    # get random features from different bins
    interesting_features_indices = []
    interesting_features_values = []
    interesting_features_category = []
    # number_features_per = 10
    for condition, condition_text in zip(conditions, conditions_texts):
        potential_indices = torch.nonzero(condition, as_tuple=True)[0]

        # Shuffle these indices and select a subset
        sampled_indices = potential_indices[torch.randperm(len(potential_indices))[:cfg.samples_per_bin]]

        values = log_freq_tokens[sampled_indices]

        interesting_features_indices = interesting_features_indices + sampled_indices.tolist()
        interesting_features_values = interesting_features_values + values.tolist()
        interesting_features_category = interesting_features_category + [f"{condition_text}"]*len(sampled_indices)

    # for v,i, c in zip(interesting_features_indices, interesting_features_values, interesting_features_category):
    #     print(c, v,i)

    print(set(interesting_features_category))
    print("Running through dataset to get top images per feature...")
    this_max = cfg.eval_max

    # max_indices = {i:None for i in interesting_features_indices}
    # max_values =  {i:None for i in interesting_features_indices} 
    # b_enc = sparse_autoencoder.b_enc[interesting_features_indices]
    # W_enc = sparse_autoencoder.W_enc[:, interesting_features_indices]

    top_per_feature = find_top_activations(val_dataloader, model, sparse_autoencoder, interesting_features_indices, [False]*len(interesting_features_indices), cfg.max_images_per_feature, this_max, cfg.batch_size, cfg.sampling_type)

    # for batch_idx, (total_images, total_labels, total_indices) in tqdm(enumerate(val_dataloader), total=this_max//cfg.batch_size): 
        # total_images = total_images.to(cfg.device)
        # total_indices = total_indices.to(cfg.device)
        # batch_size = total_images.shape[0]

        # new_top_info = highest_activating_tokens(total_images, model, sparse_autoencoder, W_enc, b_enc, interesting_features_indices) # Return all
        
        # for feature_id in interesting_features_indices:
        #     feature_data = new_top_info[feature_id]
        #     batch_image_indices = torch.tensor(feature_data['image_indices'])
        #     token_indices = torch.tensor(feature_data['token_indices'])
        #     token_activation_values = torch.tensor(feature_data['values'], device=cfg.device)
        #     global_image_indices = total_indices[batch_image_indices]  # Get global indices 

        #     # get unique image_indices
        #     # Get unique image indices and their highest activation values
        #     unique_image_indices, unique_indices = torch.unique(global_image_indices, return_inverse=True)
        #     unique_activation_values = torch.zeros_like(unique_image_indices, dtype=torch.float, device=cfg.device)
        #     unique_activation_values.index_reduce_(0, unique_indices, token_activation_values, 'amax')

        #     if max_indices[feature_id] is None: 
        #         max_indices[feature_id] = unique_image_indices
        #         max_values[feature_id] = unique_activation_values
        #     else:
        #         # Concatenate with existing data
        #         all_indices = torch.cat((max_indices[feature_id], unique_image_indices))
        #         all_values = torch.cat((max_values[feature_id], unique_activation_values))
                
        #         # Get unique indices again (in case of overlap between batches)
        #         unique_all_indices, unique_all_idx = torch.unique(all_indices, return_inverse=True)
        #         unique_all_values = torch.zeros_like(unique_all_indices, dtype=torch.float)
        #         unique_all_values.index_reduce_(0, unique_all_idx, all_values, 'amax')
                
        #         # Select top k
        #         if len(unique_all_indices) > cfg.max_images_per_feature:
        #             _, top_k_idx = torch.topk(unique_all_values, k=cfg.max_images_per_feature)
        #             max_indices[feature_id] = unique_all_indices[top_k_idx]
        #             max_values[feature_id] = unique_all_values[top_k_idx]
        #         else:
        #             max_indices[feature_id] = unique_all_indices
        #             max_values[feature_id] = unique_all_values

        # if batch_idx*cfg.batch_size >= this_max:
        #     break

    # top_per_feature = {i:(max_values[i].detach().cpu(), max_indices[i].detach().cpu()) for i in interesting_features_indices}
    ind_to_name = get_imagenet_index_to_name()

    for feature_ids, cat, logfreq in tqdm(zip(top_per_feature.keys(), interesting_features_category, interesting_features_values), total=len(interesting_features_category)):
        max_vals, max_inds = top_per_feature[feature_ids]
        images = []
        model_images = []
        gt_labels = []
        unique_bids = set()
        for bid, v in zip(max_inds, max_vals):
            if len(unique_bids) >= cfg.max_images_per_feature:
                break
            if bid not in unique_bids:
                image, label, image_ind = val_data_visualize[bid]
                images.append(image)
                model_img, _, _ = val_data[bid]
                model_images.append(model_img)
                gt_labels.append(ind_to_name[str(label)][1])
                unique_bids.add(bid)
        
        grid_size = int(np.ceil(np.sqrt(len(images))))
        fig, axs = plt.subplots(int(np.ceil(len(images)/grid_size)), grid_size, figsize=(15, 15))
        name=  f"Category: {cat},  Feature: {feature_ids}"
        fig.suptitle(name)#, y=0.95)
        for ax in axs.flatten():
            ax.axis('off')
        complete_bid = []

        for i, (image_tensor, label, val, bid, model_img) in enumerate(zip(images, gt_labels, max_vals,max_inds, model_images)):
            if bid in complete_bid:
                continue 
            complete_bid.append(bid)

            row = i // grid_size
            col = i % grid_size

            heatmap = get_heatmap(model_img,model,sparse_autoencoder, feature_ids, cfg.device)
            heatmap = image_patch_heatmap(heatmap, cfg)
            display = image_tensor.numpy().transpose(1, 2, 0)

            has_zero = False
            
            axs[row, col].imshow(display)
            axs[row, col].imshow(heatmap, cmap='viridis', alpha=0.3)  # Overlaying the heatmap
            axs[row, col].set_title(f"{label} {val.item():0.06f} {'class token!' if has_zero else ''}")  
            axs[row, col].axis('off')  

        plt.tight_layout()
        folder = os.path.join(cfg.max_image_output_folder, f"{cat}")
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, f"neglogfreq_{-logfreq}_feature_id:{feature_ids}.png"))
        # save svg
        plt.savefig(os.path.join(folder, f"neglogfreq_{-logfreq}_feature_id:{feature_ids}.svg"))
        plt.close()


if __name__ == '__main__': 

    # The argument parser will overwrite the config
    parser = argparse.ArgumentParser(description="Evaluate sparse autoencoder")
    # parser.add_argument("--sae_path", type=str, 
    #                     default='/network/scratch/s/sonia.joseph/models--Prisma-Multimodal--sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-9-hook_resid_post-l1-1e-05/snapshots/9ec4ff364c08d2413b59e8e87b55c07bd4f1096a/n_images_2600058.pt',
    #                     help="Path to sparse autoencoder")
    parser.add_argument("--model_name", type=str, 
                        default="open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",
                        help="Name of the model")
    parser.add_argument("--model_type", type=str, default="clip", help="Type of the model")
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size")
    parser.add_argument("--dataset_path", type=str, 
                        default="/network/scratch/s/sonia.joseph/datasets/kaggle_datasets",
                        help="Path to the dataset")
    parser.add_argument("--dataset_train_path", type=str, 
                        default="/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/train",
                        help="Path to the training dataset")
    parser.add_argument("--dataset_val_path", type=str, 
                        default="/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/val",
                        help="Path to the validation dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--eval_max", type=int, default=50000, help="Maximum number of samples to evaluate")
    parser.add_argument("--sampling_type", type=str, default="avg", help="Sampling type")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--samples_per_bin", type=int, default=15, help="Number of samples to collect per bin")

    total_sae_paths = [
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/a6ba5747-clip_b_mlp_out_sae_hyperparam_sweep/n_images_5200035.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/ab363ace-clip_b_mlp_out_sae_hyperparam_sweep/n_images_5200035.pt", 
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/94e3805b-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/3a32d51a-clip_b_mlp_out_sae_hyperparam_sweep/n_images_7800012.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/56d1ab26-clip_b_mlp_out_sae_hyperparam_sweep/n_images_5200035.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/ab363ace-clip_b_mlp_out_sae_hyperparam_sweep/n_images_5200035.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/bf63f410-clip_b_mlp_out_sae_hyperparam_sweep/n_images_5200035.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/db83f121-clip_b_mlp_out_sae_hyperparam_sweep/n_images_5200035.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/d907025a-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/af68edb8-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/28ec4b96-clip_b_mlp_out_sae_hyperparam_sweep/n_images_5200035.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/1a39f763-clip_b_mlp_out_sae_hyperparam_sweep/n_images_5200035.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/22dca4ec-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/9ef7b8e8-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/114f4767-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/fda9e414-clip_b_mlp_out_sae_hyperparam_sweep/n_images_5200035.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/5252bc6e-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/6a138a20-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/59e1f177-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/83c140bb-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/d38623a4-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/bd303dd0-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/223bf758-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/83c140bb-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/9c059a2a-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/ff7ab782-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/f514d05d-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/ff65f73a-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/620eb161-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/1d7618a9-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/5e3210c6-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/73ceebda-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/3fab5610-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/8a414224-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/56d59f69-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/13b6f055-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/63406ba2-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/708c90d9-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/04e3ed1b-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/65eb3a6e-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/66b0c8d7-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/d5ceb0cf-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/461cc8d6-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/4e03914d-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/04e3ed1b-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/4addbda6-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/55b1e870-clip_b_mlp_out_sae_hyperparam_sweep/n_images_2600058.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/8df4fa8f-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/b9db1c7e-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/071c1685-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/f02e630c-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/eaf339d7-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/60c3996a-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/2413e6b3-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/8b55b20a-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/3bcfb64c-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/479a7895-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/486a0773-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/d93798b6-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/dc6936e7-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/3fe6b804-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/5a049fac-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/abdf6c56-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/2f3ca24a-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/ebdc1bb3-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/38c9fccb-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/cc914f28-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/0197e57d-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/cc619a60-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/0ba5f368-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/0b57d282-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/afb41428-clip_b_mlp_out_sae_hyperparam_sweep/n_images_2600058.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/47d37623-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/4cee1e18-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/98fb20f7-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/abfba51c-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/98d43fdd-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/bed22fdf-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/d75890ba-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/ea58200a-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/eb1b4b72-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/78834fc6-clip_b_mlp_out_sae_hyperparam_sweep/n_images_2600058.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/77257359-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/5279f36d-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/a36a9d1b-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/90a626d9-clip_b_mlp_out_sae_hyperparam_sweep/n_images_2600058.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/ea58200a-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/9ea217c7-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/b7182c0e-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/25e63bd0-clip_b_mlp_out_sae_hyperparam_sweep/n_images_2600058.pt",
    "/network/scratch/s/sonia.joseph/checkpoints/clip-b/b16677fa-clip_b_mlp_out_sae_hyperparam_sweep/n_images_3900047.pt"
]

    args = parser.parse_args()


    for sae_path in total_sae_paths:
        print("Running evaluation for", sae_path)
        

        args.sae_path = sae_path
        cfg = create_eval_config(args)
        evaluate(cfg)
    
        

