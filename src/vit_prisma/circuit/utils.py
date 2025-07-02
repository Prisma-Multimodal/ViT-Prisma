from typing import List, Optional, Tuple, Union
from functools import partial
import pickle

from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_attention_mask
from einops import einsum

from .graph import Graph, AttentionNode, LogitNode


def tokenize_plus(model: HookedTransformer, inputs: List[str], max_length: Optional[int] = None):
    """
    Tokenizes the input strings using the provided model.

    Args:
        model (HookedTransformer): The model used for tokenization.
        inputs (List[str]): The list of input strings to be tokenized.

    Returns:
        tuple: A tuple containing the following elements:
            - tokens (torch.Tensor): The tokenized inputs.
            - attention_mask (torch.Tensor): The attention mask for the tokenized inputs.
            - input_lengths (torch.Tensor): The lengths of the tokenized inputs.
            - n_pos (int): The maximum sequence length of the tokenized inputs.
    """
    if max_length is not None:
        old_n_ctx = model.cfg.n_ctx
        model.cfg.n_ctx = max_length
    tokens = model.to_tokens(inputs, prepend_bos=True, padding_side='right', truncate=(max_length is not None))
    if max_length is not None:
        model.cfg.n_ctx = old_n_ctx
    attention_mask = get_attention_mask(model.tokenizer, tokens, True)
    input_lengths = attention_mask.sum(1)
    n_pos = attention_mask.size(1)
    return tokens, attention_mask, input_lengths, n_pos


def make_hooks_and_matrices(model: HookedTransformer, graph: Graph, total_items: int, batch_size: int, n_pos: int,
                            scores: Optional[Tensor], per_example_scores=None):
    """Makes a matrix, and hooks to fill it and the score matrix up

    Args:
        model (HookedTransformer): model to attribute
        graph (Graph): graph to attribute
        batch_size (int): size of the particular batch you're attributing
        n_pos (int): size of the position dimension
        scores (Tensor): The scores tensor you intend to fill. If you pass in None, we assume that you're using these hooks / matrices for evaluation only (so don't use the backwards hooks!)

    Returns:
        Tuple[Tuple[List, List, List], Tensor]: The final tensor ([batch, pos, n_src_nodes, d_model]) stores activation differences, i.e. corrupted - clean activations. The first set of hooks will add in the activations they are run on (run these on corrupted input), while the second set will subtract out the activations they are run on (run these on clean input). The third set of hooks will compute the gradients and update the scores matrix that you passed in.
    """
    # separate_activations = model.cfg.use_normalization_before_and_after and scores is None
    # if separate_activations:
    #     activation_difference = torch.zeros((2, batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=model.cfg.device, dtype=model.cfg.dtype)
    # else:
    activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model),
                                        device=model.cfg.device, dtype=model.cfg.dtype)

    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []

    # Fills up the activation difference matrix. In the default case (not separate_activations),
    # we add in the corrupted activations (add = True) and subtract out the clean ones (add=False)
    # In the separate_activations case, we just store them in two halves of the matrix. Less efficient,
    # but necessary for models with Gemma's architecture.
    def activation_hook(index, activations, hook, add: bool = True):
        acts = activations.detach()
        try:
            if add:
                activation_difference[:, :, index] += acts
            else:
                activation_difference[:, :, index] -= acts
        except RuntimeError as e:
            print(hook.name, activation_difference[:, :, index].size(), acts.size())
            raise e

    def gradient_hook(prev_index: int, bwd_index: Union[slice, int], gradients: torch.Tensor, hook):
        """Takes in a gradient and uses it and activation_difference
        to compute an update to the score matrix

        Args:
            fwd_index (Union[slice, int]): The forward index of the (ep) node
            bwd_index (Union[slice, int]): The backward index of the (dst) node
            gradients (torch.Tensor): The gradients of this backward pass
            hook (_type_): (unused)

        """
        grads = gradients.detach()
        try:
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)
            # s = einsum(activation_difference[:, :, :prev_index], grads,'batch pos forward hidden, batch pos backward hidden -> forward backward')
            s = einsum(
                activation_difference[:, :, :prev_index],
                grads,
                'batch pos forward hidden, batch pos backward hidden -> batch forward backward'
            )
            # s = s.squeeze(1)
            # scores[:prev_index, bwd_index] += s
            scores[:prev_index, bwd_index] += s.sum(dim=0).squeeze(1)  # shape: [forward, backward]
            if per_example_scores is not None:
                per_example_scores[total_items: total_items + batch_size, :prev_index, bwd_index] += s.squeeze(
                    2).detach().cpu()
        except RuntimeError as e:
            print(hook.name, activation_difference.size(), activation_difference.device, grads.size(), grads.device)
            print(prev_index, bwd_index, scores.size(), s.size())
            raise e

    node = graph.nodes['input']
    fwd_index = graph.forward_index(node)
    fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
    fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))

    for layer in range(graph.cfg['n_layers']):
        node = graph.nodes[f'a{layer}.h0']
        fwd_index = graph.forward_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        prev_index = graph.prev_index(node)
        for i, letter in enumerate('qkv'):
            bwd_index = graph.backward_index(node, qkv=letter)
            bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, prev_index, bwd_index)))

        node = graph.nodes[f'm{layer}']
        fwd_index = graph.forward_index(node)
        bwd_index = graph.backward_index(node)
        prev_index = graph.prev_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))

    node = graph.nodes['logits']
    prev_index = graph.prev_index(node)
    bwd_index = graph.backward_index(node)
    bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))

    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference


def compute_mean_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader, per_position=False,
                             device='cuda'):
    """
    Compute the mean activations of a graph's nodes over a dataset.
    """

    def activation_hook(index, activations, hook, means=None, input_lengths=None):
        # defining a hook that will fill up our means tensor. Means is of shape
        # (n_pos, graph.n_forward, model.cfg.d_model) if per_position is True, otherwise
        # (graph.n_forward, model.cfg.d_model)
        acts = activations.detach()

        # if you gave this hook input lengths, we assume you want to mean over positions
        if input_lengths is not None:
            mask = torch.zeros_like(activations)
            # mask out all padding positions
            mask[torch.arange(activations.size(0)), input_lengths - 1] = 1

            # we need ... because there might be a head index as well
            item_means = einsum(acts, mask, 'batch pos ... hidden, batch pos ... hidden -> batch ... hidden')

            # mean over the positions we did take, position-wise
            if len(item_means.size()) == 3:
                item_means /= input_lengths.unsqueeze(-1).unsqueeze(-1)
            else:
                item_means /= input_lengths.unsqueeze(-1)

            means[index] += item_means.sum(0)
        else:
            means[:, index] += acts.sum(0)

    # we're going to get all of the out hooks / indices we need for making hooks
    # but we can't make them until we have input length masks
    processed_attn_layers = set()
    hook_points_indices = []
    for node in graph.nodes.values():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            processed_attn_layers.add(node.layer)

        if not isinstance(node, LogitNode):
            hook_points_indices.append((node.out_hook, graph.forward_index(node)))

    means_initialized = False
    total = 0
    for batch in tqdm(dataloader, desc='Computing mean'):
        # maybe the dataset is given as a tuple, maybe its just raw strings
        clean = batch[0] if isinstance(batch, tuple) else batch
        clean_images = torch.stack(clean).to(device)
        total += len(clean_images)

        if not means_initialized:
            # here is where we store the means
            if per_position:
                means = torch.zeros(
                    ((model.cfg.image_size // model.cfg.patch_size) ** 2 + 1, graph.n_forward, model.cfg.d_model),
                    device='cuda', dtype=model.cfg.dtype)
            else:
                means = torch.zeros((graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)
            means_initialized = True

        if per_position:
            input_lengths = None
        else:
            input_lengths = torch.tensor([0 for i in range(len(clean))])
        add_to_mean_hooks = [(hook_point, partial(activation_hook, index, means=means, input_lengths=input_lengths)) for
                             hook_point, index in hook_points_indices]

        with model.hooks(fwd_hooks=add_to_mean_hooks):
            model(clean_images)

    means = means.squeeze(0)
    means /= total
    return means if per_position else means.mean(0)


def load_ablations(model: HookedTransformer, graph: Graph, ablation_path: str):
    """
    Load pre-computed activations to be used for ablations.
    Expects a pickle file containing a dictionary of the format {'block_name': Tensor[(1,) d_component_output]}
    """
    with open(ablation_path, "rb") as handle:
        activations_precomp = pickle.load(handle)

    activations_optimal = torch.zeros((graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)

    processed_attn_layers = set()
    for node in graph.nodes.values():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            processed_attn_layers.add(node.layer)

        if not isinstance(node, LogitNode):
            if node.name == "input":
                activations_optimal[0] = activations_precomp["hook_embed"]
            elif node.name.startswith("a"):
                head_idx = int(node.name.split("h")[-1])
                acts_idx = (model.cfg.n_heads + 1) * node.layer + head_idx + 1
                activations_optimal[acts_idx] = activations_precomp[f"blocks.{node.layer}.attn.hook_result"][head_idx]
            elif node.name.startswith("m"):
                acts_idx = (model.cfg.n_heads + 1) * node.layer + model.cfg.n_heads + 1
                activations_optimal[acts_idx] = activations_precomp[f"blocks.{node.layer}.hook_mlp_out"].squeeze()

    return activations_optimal