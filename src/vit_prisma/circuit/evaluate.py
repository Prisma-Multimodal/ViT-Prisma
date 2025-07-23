import math
import os
import pickle
from typing import Callable, List, Union, Literal, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import einsum

from .utils import tokenize_plus, make_hooks_and_matrices, compute_mean_activations, load_ablations
from .graph import Graph, AttentionNode
from ..models.base_vit import HookedViT


def evaluate_graph(model: HookedViT, graph: Graph, dataloader: DataLoader,
                   metrics: Union[Callable[[Tensor], Tensor], List[Callable[[Tensor], Tensor]]],
                   quiet=False, device='cuda', task=None, model_name=None, model_id=None,
                   intervention: Literal['patching', 'zero', 'mean', 'mean-positional', 'optimal'] = 'patching',
                   intervention_dataloader: Optional[DataLoader] = None, optimal_ablation_path: Optional[str] = None,
                   skip_clean: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Evaluate a circuit (i.e. a graph where only some nodes are false, probably created by calling graph.apply_threshold). You probably want to prune beforehand to make sure your circuit is valid.

    Args:
        model (HookedViT): The model to run the circuit on
        graph (Graph): The circuit to evaluate
        dataloader (DataLoader): The dataset to evaluate on
        metrics (Union[Callable[[Tensor],Tensor], List[Callable[[Tensor], Tensor]]]): The metric(s) to evaluate with respect to
        quiet (bool, optional): Whether to silence the tqdm progress bar. Defaults to False.
        intervention (Literal[&#39;patching&#39;, &#39;zero&#39;, &#39;mean&#39;,&#39;mean, optional): Which ablation to evaluate with respect to. 'patching' is an interchange intervention; mean-positional takes the positional mean over the given dataset. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): The dataset to take the mean over. Must be set if intervention is mean or mean-positional. Defaults to None.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: A tensor (or list thereof) of faithfulness scores; if a list, each list entry corresponds to a metric in the input list
    """
    assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"

    assert intervention in ['patching', 'zero', 'mean', 'mean-positional'], f"Invalid intervention: {intervention}"

    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        if model_id is not None:
            model_name = f'waterbirds_sweep_{model_id}'
        if os.path.exists(f'activations/{task}_{model_name}_{intervention}.p'):
            with open(f'activations/{task}_{model_name}_{intervention}.p', 'rb') as file:
                means = pickle.load(file)
        else:
            per_position = 'positional' in intervention
            means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position,
                                             device=device)
            means = means.unsqueeze(0)
            if not per_position:
                means = means.unsqueeze(0)
            with open(f'activations/{task}_{model_name}_{intervention}.p', 'wb') as file:
                pickle.dump(means, file)

    elif intervention == 'optimal':
        assert optimal_ablation_path is not None, "Path to pre-computed activations must be provided for optimal ablations"
        optimal_ablations = load_ablations(model, graph, optimal_ablation_path)
        optimal_ablations = optimal_ablations.unsqueeze(0).unsqueeze(0)

    # This step cleans up the graph, removing components until it's fully connected
    print(f'before prune: {graph.weighted_edge_count()}')
    graph.prune()
    print(f'after prune: {graph.weighted_edge_count()}')

    # Construct a matrix that indicates which edges are in the graph
    in_graph_matrix = graph.in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)

    # same thing but for neurons
    if graph.neurons_in_graph is not None:
        neuron_matrix = graph.neurons_in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)

        # If an edge is in the graph, but not all its neurons are, we need to update that edge anyway
        node_fully_in_graph = (neuron_matrix.sum(-1) == model.cfg.d_model).to(model.cfg.dtype)
        in_graph_matrix = einsum(in_graph_matrix, node_fully_in_graph, 'forward backward, forward -> forward backward')
    else:
        neuron_matrix = None

    # We take the opposite matrix, because we'll use it as a mask to specify
    # which edges we want to corrupt
    in_graph_matrix = 1 - in_graph_matrix
    if neuron_matrix is not None:
        neuron_matrix = 1 - neuron_matrix

    if model.cfg.use_normalization_before_and_after:
        # If the model also normalizes the outputs of attention heads, we'll need to take that into account when evaluating the graph.
        attention_head_mask = torch.zeros((graph.n_forward, model.cfg.n_layers), device='cuda', dtype=model.cfg.dtype)
        for node in graph.nodes.values():
            if isinstance(node, AttentionNode):
                attention_head_mask[graph.forward_index(node), node.layer] = 1

        non_attention_head_mask = 1 - attention_head_mask.any(-1).to(dtype=model.cfg.dtype)
        attention_biases = torch.stack([block.attn.b_O for block in model.blocks])

    # For each node in the graph, corrupt its inputs, if the corresponding edge isn't in the graph
    # We corrupt it by adding in the activation difference (b/w clean and corrupted acts)
    def make_input_construction_hook(activation_matrix, in_graph_vector, neuron_matrix):
        def input_construction_hook(activations, hook):
            # Case where layernorm is applied after attention (gemma only)
            if model.cfg.use_normalization_before_and_after:
                activation_differences = activation_matrix[0] - activation_matrix[1]

                # get the clean outputs of the attention heads that came before
                clean_attention_results = einsum(activation_matrix[1, :, :, :len(in_graph_vector)],
                                                 attention_head_mask[:len(in_graph_vector)],
                                                 'batch pos previous hidden, previous layer -> batch pos layer hidden')

                # get the update corresponding to non-attention heads, and the difference between clean and corrupted attention heads
                if neuron_matrix is not None:
                    non_attention_update = einsum(activation_differences[:, :, :len(in_graph_vector)],
                                                  neuron_matrix[:len(in_graph_vector)],
                                                  in_graph_vector,
                                                  non_attention_head_mask[:len(in_graph_vector)],
                                                  'batch pos previous hidden, previous hidden, previous ..., previous -> batch pos ... hidden')
                    corrupted_attention_difference = einsum(activation_differences[:, :, :len(in_graph_vector)],
                                                            neuron_matrix[:len(in_graph_vector)],
                                                            in_graph_vector,
                                                            attention_head_mask[:len(in_graph_vector)],
                                                            'batch pos previous hidden, previous hidden, previous ..., previous layer -> batch pos ... layer hidden')
                else:
                    non_attention_update = einsum(activation_differences[:, :, :len(in_graph_vector)],
                                                  in_graph_vector,
                                                  non_attention_head_mask[:len(in_graph_vector)],
                                                  'batch pos previous hidden, previous ..., previous -> batch pos ... hidden')
                    corrupted_attention_difference = einsum(activation_differences[:, :, :len(in_graph_vector)],
                                                            in_graph_vector,
                                                            attention_head_mask[:len(in_graph_vector)],
                                                            'batch pos previous hidden, previous ..., previous layer -> batch pos ... layer hidden')

                # add the biases to the attention results, and compute the corrupted attention results using the difference
                # we process all the attention heads at once; this is how we can tell if we're doing that
                if in_graph_vector.ndim == 2:
                    corrupted_attention_results = clean_attention_results.unsqueeze(2) + corrupted_attention_difference
                    # (1, 1, 1, layer, hidden)
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                else:
                    corrupted_attention_results = clean_attention_results + corrupted_attention_difference
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)

                # pass both the clean and corrupted attention results through the layernorm and
                # add the difference to the update
                update = non_attention_update
                valid_layers = attention_head_mask[:len(in_graph_vector)].any(0)
                for i, valid_layer in enumerate(valid_layers):
                    if not valid_layer:
                        break
                    if in_graph_vector.ndim == 2:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, None, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, :, i])
                    else:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, i])

            else:
                # In the non-gemma case, things are easy!
                activation_differences = activation_matrix
                # The ... here is to account for a potential head dimension, when constructing a whole attention layer's input
                if neuron_matrix is not None:
                    update = einsum(activation_differences[:, :, :len(in_graph_vector)],
                                    neuron_matrix[:len(in_graph_vector)], in_graph_vector,
                                    'batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden')
                else:
                    update = einsum(activation_differences[:, :, :len(in_graph_vector)], in_graph_vector,
                                    'batch pos previous hidden, previous ... -> batch pos ... hidden')
            activations += update
            return activations

        return input_construction_hook

    def make_input_construction_hooks(activation_differences, in_graph_matrix, neuron_matrix):
        input_construction_hooks = []
        for layer in range(model.cfg.n_layers):
            # If any attention node in the layer is in the graph, just construct the input for the entire layer
            if any(graph.nodes[f'a{layer}.h{head}'].in_graph for head in range(model.cfg.n_heads)) and \
                    not (neuron_matrix is None and all(
                        parent_edge.in_graph for head in range(model.cfg.n_heads) for parent_edge in
                        graph.nodes[f'a{layer}.h{head}'].parent_edges)):
                for i, letter in enumerate('qkv'):
                    node = graph.nodes[f'a{layer}.h0']
                    prev_index = graph.prev_index(node)
                    bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True)
                    input_cons_hook = make_input_construction_hook(activation_differences,
                                                                   in_graph_matrix[:prev_index, bwd_index],
                                                                   neuron_matrix)
                    input_construction_hooks.append((node.qkv_inputs[i], input_cons_hook))

            # add MLP hook if MLP in graph
            if graph.nodes[f'm{layer}'].in_graph and \
                    not (neuron_matrix is None and all(
                        parent_edge.in_graph for parent_edge in graph.nodes[f'm{layer}'].parent_edges)):
                node = graph.nodes[f'm{layer}']
                prev_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node)
                input_cons_hook = make_input_construction_hook(activation_differences,
                                                               in_graph_matrix[:prev_index, bwd_index], neuron_matrix)
                input_construction_hooks.append((node.in_hook, input_cons_hook))

        # Always add the logits hook
        if not (neuron_matrix is None and all(
                parent_edge.in_graph for parent_edge in graph.nodes['logits'].parent_edges)):
            node = graph.nodes['logits']
            fwd_index = graph.prev_index(node)
            bwd_index = graph.backward_index(node)
            input_cons_hook = make_input_construction_hook(activation_differences,
                                                           in_graph_matrix[:fwd_index, bwd_index], neuron_matrix)
            input_construction_hooks.append((node.in_hook, input_cons_hook))

        return input_construction_hooks

    # convert metrics to list if it's not already
    if not isinstance(metrics, list):
        metrics = [metrics]
    results = [[] for _ in metrics]

    # and here we actually run / evaluate the model
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        clean_images = clean.to(device)
        if corrupted[0] is not None:
            corrupted_images = corrupted.to(device)

        # fwd_hooks_corrupted adds in corrupted acts to activation_difference
        # fwd_hooks_clean subtracts out clean acts from activation_difference
        # activation difference is of size (batch, pos, src_nodes, hidden)
        (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = make_hooks_and_matrices(model, graph, 0,
                                                                                                   len(clean), (
                                                                                                               model.cfg.image_size // model.cfg.patch_size) ** 2 + 1,
                                                                                                   None)

        input_construction_hooks = make_input_construction_hooks(activation_difference, in_graph_matrix, neuron_matrix)
        with torch.inference_mode():
            if intervention == 'patching':
                # We intervene by subtracting out clean and adding in corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    corrupted_logits = model(corrupted_images)
            else:
                # In the case of zero or mean ablation, we skip the adding in corrupted activations
                # but in mean ablations, we need to add the mean in
                if 'mean' in intervention:
                    activation_difference += means

                if intervention == 'optimal':
                    activation_difference += optimal_ablations

            # For some metrics (e.g. accuracy or KL), we need the clean logits
            clean_logits = None if skip_clean else model(clean_images)

            with model.hooks(fwd_hooks_clean + input_construction_hooks):
                logits = model(clean_images)

        for i, metric in enumerate(metrics):
            r = metric(logits, clean_logits, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    # unwrap the results if there's only one metric
    if len(results) == 1:
        results = results[0]
    return results


def evaluate_graph_per_sample(model: HookedViT, graph: Graph, clean, corrupted, label,
                              metrics: Union[Callable[[Tensor], Tensor], List[Callable[[Tensor], Tensor]]],
                              quiet=False,
                              intervention: Literal[
                                  'patching', 'zero', 'mean', 'mean-positional', 'optimal'] = 'patching',
                              intervention_dataloader: Optional[DataLoader] = None,
                              optimal_ablation_path: Optional[str] = None,
                              skip_clean: bool = True, device='cuda') -> Union[torch.Tensor, List[torch.Tensor]]:
    """Evaluate a circuit (i.e. a graph where only some nodes are false, probably created by calling graph.apply_threshold). You probably want to prune beforehand to make sure your circuit is valid.

    Args:
        model (HookedTransformer): The model to run the circuit on
        graph (Graph): The circuit to evaluate
        dataloader (DataLoader): The dataset to evaluate on
        metrics (Union[Callable[[Tensor],Tensor], List[Callable[[Tensor], Tensor]]]): The metric(s) to evaluate with respect to
        quiet (bool, optional): Whether to silence the tqdm progress bar. Defaults to False.
        intervention (Literal[&#39;patching&#39;, &#39;zero&#39;, &#39;mean&#39;,&#39;mean, optional): Which ablation to evaluate with respect to. 'patching' is an interchange intervention; mean-positional takes the positional mean over the given dataset. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): The dataset to take the mean over. Must be set if intervention is mean or mean-positional. Defaults to None.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: A tensor (or list thereof) of faithfulness scores; if a list, each list entry corresponds to a metric in the input list
    """
    assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"

    assert intervention in ['patching', 'zero', 'mean', 'mean-positional'], f"Invalid intervention: {intervention}"

    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    elif intervention == 'optimal':
        assert optimal_ablation_path is not None, "Path to pre-computed activations must be provided for optimal ablations"
        optimal_ablations = load_ablations(model, graph, optimal_ablation_path)
        optimal_ablations = optimal_ablations.unsqueeze(0).unsqueeze(0)

    # This step cleans up the graph, removing components until it's fully connected
    graph.prune()

    # Construct a matrix that indicates which edges are in the graph
    in_graph_matrix = graph.in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)

    # same thing but for neurons
    if graph.neurons_in_graph is not None:
        neuron_matrix = graph.neurons_in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)

        # If an edge is in the graph, but not all its neurons are, we need to update that edge anyway
        node_fully_in_graph = (neuron_matrix.sum(-1) == model.cfg.d_model).to(model.cfg.dtype)
        in_graph_matrix = einsum(in_graph_matrix, node_fully_in_graph,
                                 'forward backward, forward -> forward backward')
    else:
        neuron_matrix = None

    # We take the opposite matrix, because we'll use it as a mask to specify
    # which edges we want to corrupt
    in_graph_matrix = 1 - in_graph_matrix
    if neuron_matrix is not None:
        neuron_matrix = 1 - neuron_matrix

    if model.cfg.use_normalization_before_and_after:
        # If the model also normalizes the outputs of attention heads, we'll need to take that into account when evaluating the graph.
        attention_head_mask = torch.zeros((graph.n_forward, model.cfg.n_layers), device='cuda',
                                          dtype=model.cfg.dtype)
        for node in graph.nodes.values():
            if isinstance(node, AttentionNode):
                attention_head_mask[graph.forward_index(node), node.layer] = 1

        non_attention_head_mask = 1 - attention_head_mask.any(-1).to(dtype=model.cfg.dtype)
        attention_biases = torch.stack([block.attn.b_O for block in model.blocks])

    # For each node in the graph, corrupt its inputs, if the corresponding edge isn't in the graph
    # We corrupt it by adding in the activation difference (b/w clean and corrupted acts)
    def make_input_construction_hook(activation_matrix, in_graph_vector, neuron_matrix):
        def input_construction_hook(activations, hook):
            # Case where layernorm is applied after attention (gemma only)
            if model.cfg.use_normalization_before_and_after:
                activation_differences = activation_matrix[0] - activation_matrix[1]

                # get the clean outputs of the attention heads that came before
                clean_attention_results = einsum(activation_matrix[1, :, :, :len(in_graph_vector)],
                                                 attention_head_mask[:len(in_graph_vector)],
                                                 'batch pos previous hidden, previous layer -> batch pos layer hidden')

                # get the update corresponding to non-attention heads, and the difference between clean and corrupted attention heads
                if neuron_matrix is not None:
                    non_attention_update = einsum(activation_differences[:, :, :len(in_graph_vector)],
                                                  neuron_matrix[:len(in_graph_vector)],
                                                  in_graph_vector,
                                                  non_attention_head_mask[:len(in_graph_vector)],
                                                  'batch pos previous hidden, previous hidden, previous ..., previous -> batch pos ... hidden')
                    corrupted_attention_difference = einsum(activation_differences[:, :, :len(in_graph_vector)],
                                                            neuron_matrix[:len(in_graph_vector)],
                                                            in_graph_vector,
                                                            attention_head_mask[:len(in_graph_vector)],
                                                            'batch pos previous hidden, previous hidden, previous ..., previous layer -> batch pos ... layer hidden')
                else:
                    non_attention_update = einsum(activation_differences[:, :, :len(in_graph_vector)],
                                                  in_graph_vector,
                                                  non_attention_head_mask[:len(in_graph_vector)],
                                                  'batch pos previous hidden, previous ..., previous -> batch pos ... hidden')
                    corrupted_attention_difference = einsum(activation_differences[:, :, :len(in_graph_vector)],
                                                            in_graph_vector,
                                                            attention_head_mask[:len(in_graph_vector)],
                                                            'batch pos previous hidden, previous ..., previous layer -> batch pos ... layer hidden')

                # add the biases to the attention results, and compute the corrupted attention results using the difference
                # we process all the attention heads at once; this is how we can tell if we're doing that
                if in_graph_vector.ndim == 2:
                    corrupted_attention_results = clean_attention_results.unsqueeze(
                        2) + corrupted_attention_difference
                    # (1, 1, 1, layer, hidden)
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                else:
                    corrupted_attention_results = clean_attention_results + corrupted_attention_difference
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)

                # pass both the clean and corrupted attention results through the layernorm and
                # add the difference to the update
                update = non_attention_update
                valid_layers = attention_head_mask[:len(in_graph_vector)].any(0)
                for i, valid_layer in enumerate(valid_layers):
                    if not valid_layer:
                        break
                    if in_graph_vector.ndim == 2:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, None, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, :, i])
                    else:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, i])

            else:
                # In the non-gemma case, things are easy!
                activation_differences = activation_matrix
                # The ... here is to account for a potential head dimension, when constructing a whole attention layer's input
                if neuron_matrix is not None:
                    update = einsum(activation_differences[:, :, :len(in_graph_vector)],
                                    neuron_matrix[:len(in_graph_vector)], in_graph_vector,
                                    'batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden')
                else:
                    update = einsum(activation_differences[:, :, :len(in_graph_vector)], in_graph_vector,
                                    'batch pos previous hidden, previous ... -> batch pos ... hidden')
            activations += update
            return activations

        return input_construction_hook

    def make_input_construction_hooks(activation_differences, in_graph_matrix, neuron_matrix):
        input_construction_hooks = []
        for layer in range(model.cfg.n_layers):
            # If any attention node in the layer is in the graph, just construct the input for the entire layer
            if any(graph.nodes[f'a{layer}.h{head}'].in_graph for head in range(model.cfg.n_heads)) and \
                    not (neuron_matrix is None and all(
                        parent_edge.in_graph for head in range(model.cfg.n_heads) for parent_edge in
                        graph.nodes[f'a{layer}.h{head}'].parent_edges)):
                for i, letter in enumerate('qkv'):
                    node = graph.nodes[f'a{layer}.h0']
                    prev_index = graph.prev_index(node)
                    bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True)
                    input_cons_hook = make_input_construction_hook(activation_differences,
                                                                   in_graph_matrix[:prev_index, bwd_index],
                                                                   neuron_matrix)
                    input_construction_hooks.append((node.qkv_inputs[i], input_cons_hook))

            # add MLP hook if MLP in graph
            if graph.nodes[f'm{layer}'].in_graph and \
                    not (neuron_matrix is None and all(
                        parent_edge.in_graph for parent_edge in graph.nodes[f'm{layer}'].parent_edges)):
                node = graph.nodes[f'm{layer}']
                prev_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node)
                input_cons_hook = make_input_construction_hook(activation_differences,
                                                               in_graph_matrix[:prev_index, bwd_index], neuron_matrix)
                input_construction_hooks.append((node.in_hook, input_cons_hook))

        # Always add the logits hook
        if not (neuron_matrix is None and all(
                parent_edge.in_graph for parent_edge in graph.nodes['logits'].parent_edges)):
            node = graph.nodes['logits']
            fwd_index = graph.prev_index(node)
            bwd_index = graph.backward_index(node)
            input_cons_hook = make_input_construction_hook(activation_differences,
                                                           in_graph_matrix[:fwd_index, bwd_index], neuron_matrix)
            input_construction_hooks.append((node.in_hook, input_cons_hook))

        return input_construction_hooks

    # convert metrics to list if it's not already
    if not isinstance(metrics, list):
        metrics = [metrics]
    results = [[] for _ in metrics]

    # and here we actually run / evaluate the model

    # fwd_hooks_corrupted adds in corrupted acts to activation_difference
    # fwd_hooks_clean subtracts out clean acts from activation_difference
    # activation difference is of size (batch, pos, src_nodes, hidden)
    (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = make_hooks_and_matrices(model, graph, len(clean),
                                                                                               (
                                                                                                           model.cfg.image_size // model.cfg.patch_size) ** 2 + 1,
                                                                                               None)

    input_construction_hooks = make_input_construction_hooks(activation_difference, in_graph_matrix, neuron_matrix)
    with torch.inference_mode():
        if intervention == 'patching':
            # We intervene by subtracting out clean and adding in corrupted activations
            with model.hooks(fwd_hooks_corrupted):
                corrupted_logits = model(corrupted)
        else:
            # In the case of zero or mean ablation, we skip the adding in corrupted activations
            # but in mean ablations, we need to add the mean in
            if 'mean' in intervention:
                activation_difference += means

            if intervention == 'optimal':
                activation_difference += optimal_ablations

        # For some metrics (e.g. accuracy or KL), we need the clean logits
        clean_logits = None if skip_clean else model(clean)

        with model.hooks(fwd_hooks_clean + input_construction_hooks):
            logits = model(clean)

    for i, metric in enumerate(metrics):
        r = metric(logits, clean_logits, label).cpu()
        if len(r.size()) == 0:
            r = r.unsqueeze(0)
        results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    # unwrap the results if there's only one metric
    if len(results) == 1:
        results = results[0]
    return results


def evaluate_baseline(model: HookedViT, dataloader: DataLoader, metrics: List[Callable[[Tensor], Tensor]],
                      run_corrupted=False, quiet=False, device='cuda') -> Union[torch.Tensor, List[torch.Tensor]]:
    """Evaluates the model on the given dataloader, without any intervention. This is useful for computing the baseline performance of the model.

    Args:
        model (HookedTransformer): The model to evaluate
        dataloader (DataLoader): The dataset to evaluate on
        metrics (List[Callable[[Tensor], Tensor]]): The metrics to evaluate with respect to
        run_corrupted (bool, optional): Whether to evaluate on corrupted examples instead. Defaults to False.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: A tensor (or list thereof) of performance scores; if a list, each list entry corresponds to a metric in the input list
    """
    if not isinstance(metrics, list):
        metrics = [metrics]

    results = [[] for _ in metrics]
    if not quiet:
        dataloader = tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        clean_images = clean.to(device)
        if corrupted[0] is not None:
            corrupted_images = corrupted.to(device)
        with torch.inference_mode():
            if corrupted[0] is not None:
                corrupted_logits = model(corrupted_images)
            else:
                corrupted_logits = None
            logits = model(clean_images)
        for i, metric in enumerate(metrics):
            if run_corrupted:
                r = metric(corrupted_logits, logits, label).cpu()
            else:
                r = metric(logits, corrupted_logits, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if len(results) == 1:
        results = results[0]
    return results


def evaluate_baseline_per_sample(model: HookedViT, clean_input, corrupted_input, label,
                                 metrics: List[Callable[[Tensor], Tensor]],
                                 run_corrupted=False, quiet=False, device='cuda') -> Union[
    torch.Tensor, List[torch.Tensor]]:
    """Evaluates the model on the given data sample, without any intervention. This is useful for computing the baseline performance of the model.

    Args:
        model (HookedTransformer): The model to evaluate
        dataloader (DataLoader): The dataset to evaluate on
        metrics (List[Callable[[Tensor], Tensor]]): The metrics to evaluate with respect to
        run_corrupted (bool, optional): Whether to evaluate on corrupted examples instead. Defaults to False.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: A tensor (or list thereof) of performance scores; if a list, each list entry corresponds to a metric in the input list
    """
    if not isinstance(metrics, list):
        metrics = [metrics]

    results = [[] for _ in metrics]
    with torch.inference_mode():
        corrupted_logits = model(corrupted_input)
        logits = model(clean_input)
    for i, metric in enumerate(metrics):
        if run_corrupted:
            r = metric(corrupted_logits, logits, label).cpu()
        else:
            r = metric(logits, corrupted_logits, label).cpu()
        if len(r.size()) == 0:
            r = r.unsqueeze(0)
        results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if len(results) == 1:
        results = results[0]
    return results


def evaluate_area_under_curve(model: HookedViT, graph: Graph, dataloader, metrics, quiet: bool = False,
                              level: Literal['edge', 'node', 'neuron'] = 'edge', log_scale: bool = False,
                              absolute: bool = True, task=None, model_name=None,
                              intervention: Literal['patching', 'zero', 'mean', 'mean-positional'] = 'patching',
                              intervention_dataloader: DataLoader = None,
                              optimal_ablation_path: Optional[str] = None, no_normalize: Optional[bool] = False,
                              apply_greedy: bool = False):
    baseline_score = evaluate_baseline(model, dataloader, metrics).mean().item()
    graph.apply_topn(0, True)
    corrupted_score = evaluate_graph(model, graph, dataloader, metrics, quiet=quiet, intervention=intervention,
                                     intervention_dataloader=intervention_dataloader, task=task,
                                     model_name=model_name).mean().item()

    if level == 'neuron':
        assert graph.neurons_scores is not None, "Neuron scores must be present for neuron-level evaluation"
        n_scored_items = (~torch.isnan(graph.neurons_scores)).sum().item()
    elif level == 'node':
        assert graph.nodes_scores is not None, "Node scores must be present for node-level evaluation"
        n_scored_items = (~torch.isnan(graph.nodes_scores)).sum().item()
    else:
        n_scored_items = len(graph.edges)

    percentages = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)

    faithfulnesses = []
    weighted_edge_counts = []
    for pct in percentages:
        this_graph = graph
        curr_num_items = int(pct * n_scored_items)
        print(f"Computing results for {pct * 100}% of {level}s (N={curr_num_items})")
        if apply_greedy:
            assert level == 'edge', "Greedy application only supported for edge-level evaluation"
            this_graph.apply_greedy(curr_num_items, absolute=absolute, prune=True)
        else:
            this_graph.apply_topn(curr_num_items, absolute, level=level, prune=True)

        weighted_edge_count = this_graph.weighted_edge_count()
        weighted_edge_counts.append(weighted_edge_count + 1 if weighted_edge_count == 0 else weighted_edge_count)

        ablated_score = evaluate_graph(model, this_graph, dataloader, metrics,
                                       quiet=quiet, intervention=intervention,
                                       intervention_dataloader=intervention_dataloader, task=task,
                                       model_name=model_name).mean().item()
        if no_normalize:
            faithfulness = ablated_score
        else:
            faithfulness = (ablated_score - corrupted_score) / (baseline_score - corrupted_score)
        faithfulnesses.append(faithfulness)

    area_under = 0.
    area_from_1 = 0.
    for i in range(len(faithfulnesses) - 1):
        i_1, i_2 = i, i + 1
        x_1 = percentages[i_1]
        x_2 = percentages[i_2]
        # area from point to 100
        if log_scale:
            x_1 = math.log(x_1)
            x_2 = math.log(x_2)
        trapezoidal = (x_2 - x_1) * \
                      (((abs(1. - faithfulnesses[i_1])) + (abs(1. - faithfulnesses[i_2]))) / 2)
        area_from_1 += trapezoidal

        trapezoidal = (x_2 - x_1) * ((faithfulnesses[i_1] + faithfulnesses[i_2]) / 2)
        area_under += trapezoidal
    average = sum(faithfulnesses) / len(faithfulnesses)
    return weighted_edge_counts, percentages, area_under, area_from_1, average, faithfulnesses