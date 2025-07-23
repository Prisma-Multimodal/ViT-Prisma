import os.path
import pickle
from typing import Callable, List, Union, Optional, Literal, Tuple
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer

from tqdm import tqdm

from .utils import tokenize_plus, make_hooks_and_matrices, compute_mean_activations, load_ablations
from .evaluate import evaluate_graph, evaluate_graph_per_sample, evaluate_baseline, evaluate_baseline_per_sample
from .evaluate import Graph

def get_scores_exact(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor],
                     intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching',
                     intervention_dataloader: Optional[DataLoader]=None, optimal_ablation_path: Optional[str] = None, quiet=False, device='cuda'):
    """Gets scores via exact patching, by repeatedly calling evaluate graph.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        intervention (Literal[&#39;patching&#39;, &#39;zero&#39;, &#39;mean&#39;,&#39;mean, optional): the intervention to use. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): the dataloader over which to take the mean. Defaults to None.
        quiet (bool, optional): _description_. Defaults to False.
    """

    graph.in_graph |= graph.real_edge_mask  # All edges that are real are now in the graph
    baseline = evaluate_baseline(model, dataloader, metric, device=device).mean().item()
    edges = graph.edges.values() if quiet else tqdm(graph.edges.values())
    for edge in edges:
        edge.in_graph = False
        intervened_performance = evaluate_graph(model, graph, dataloader, metric, intervention=intervention,
                                                intervention_dataloader=intervention_dataloader,
                                                optimal_ablation_path=optimal_ablation_path, quiet=True,
                                                skip_clean=True, device=device).mean().item()
        edge.score = intervened_performance - baseline
        edge.in_graph = True

    # This is just to make the return type the same as all of the others; we've actually already updated the score matrix
    return graph.scores

def get_scores_exact_optimized(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor],
                     intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching',
                     intervention_dataloader: Optional[DataLoader]=None, optimal_ablation_path: Optional[str] = None, quiet=False, device='cuda'):
    """Gets scores via exact patching, by repeatedly calling evaluate graph.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        intervention (Literal[&#39;patching&#39;, &#39;zero&#39;, &#39;mean&#39;,&#39;mean, optional): the intervention to use. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): the dataloader over which to take the mean. Defaults to None.
        quiet (bool, optional): _description_. Defaults to False.
    """
    num_samples = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        clean = clean.to(device)
        corrupted = corrupted.to(device)
        graph.in_graph |= graph.real_edge_mask  # All edges that are real are now in the graph
        baseline = evaluate_baseline_per_sample(model, clean, corrupted, label, metric, device=device).mean().item()
        edges = graph.edges.values()
        for edge in tqdm(edges):
            edge.in_graph = False
            intervened_performance = evaluate_graph_per_sample(model, graph, clean, corrupted, label, metric, intervention=intervention,
                                                    intervention_dataloader=intervention_dataloader,
                                                    optimal_ablation_path=optimal_ablation_path, quiet=True,
                                                    skip_clean=True, device=device).mean().item()
            edge.score += intervened_performance - baseline
            edge.in_graph = True
        num_samples += 1
    edges = graph.edges.values()
    for edge in edges:
        edge.score /= num_samples

    # This is just to make the return type the same as all of the others; we've actually already updated the score matrix
    return graph.scores

def get_scores_exact_optimized_parallel(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor],
                     intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching',
                     intervention_dataloader: Optional[DataLoader]=None, optimal_ablation_path: Optional[str] = None, quiet=False, device='cuda'):
    """Gets scores via exact patching, by repeatedly calling evaluate graph.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        intervention (Literal[&#39;patching&#39;, &#39;zero&#39;, &#39;mean&#39;,&#39;mean, optional): the intervention to use. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): the dataloader over which to take the mean. Defaults to None.
        quiet (bool, optional): _description_. Defaults to False.
    """
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        clean = clean.to(device)
        corrupted = corrupted.to(device)
        graph.in_graph |= graph.real_edge_mask  # All edges that are real are now in the graph
        baseline = evaluate_baseline_per_sample(model, clean, corrupted, label, metric, device=device).mean().item()
        edges = graph.edges.values()
        for edge in tqdm(edges):
            edge.in_graph = False
            intervened_performance = evaluate_graph_per_sample(model, graph, clean, corrupted, label, metric, intervention=intervention,
                                                    intervention_dataloader=intervention_dataloader,
                                                    optimal_ablation_path=optimal_ablation_path, quiet=True,
                                                    skip_clean=True, device=device).mean().item()
            edge.score += intervened_performance - baseline
            edge.in_graph = True

    # This is just to make the return type the same as all of the others; we've actually already updated the score matrix
    return graph.scores


def get_scores_eap(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
                   intervention: Literal['patching', 'zero', 'mean', 'mean-positional', 'optimal'] = 'patching',
                   intervention_dataloader: Optional[DataLoader] = None, optimal_ablation_path: Optional[str] = None,
                   quiet=False, device='cuda'):
    """Gets edge attribution scores using EAP.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)

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

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_images = clean.to(device)
        corrupted_images = corrupted.to(device)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, total_items - batch_size,
                                                                                                           batch_size,
                                                                                                           (model.cfg.image_size // model.cfg.patch_size)**2+1,
                                                                                                           scores)

        with torch.inference_mode():
            if intervention == 'patching':
                # We intervene by subtracting out clean and adding in corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    _ = model(corrupted_images)
            elif 'mean' in intervention:
                # In the case of zero or mean ablation, we skip the adding in corrupted activations
                # but in mean ablations, we need to add the mean in
                activation_difference += means

            elif intervention == 'optimal':
                activation_difference += optimal_ablations

            # For some metrics (e.g. accuracy or KL), we need the clean logits
            clean_logits = model(clean_images)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean_images)
            metric_value = metric(logits, clean_logits, label)
            metric_value.backward()

    scores /= total_items

    return scores

def get_scores_eap_ig(model: HookedTransformer, graph: Graph, dataloader: DataLoader,
                      metric: Callable[[Tensor], Tensor], intervention: Literal[
            'patching', 'zero', 'mean', 'mean-positional', 'optimal'] = 'patching',
                      steps=30, intervention_dataloader: Optional[DataLoader] = None,
                      optimal_ablation_path: Optional[str] = None, quiet=False, device='cuda', task=None, model_name=None, model_id=None, get_perexample_scores=False):
    """Gets edge attribution scores using EAP with integrated gradients.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        steps (int, optional): number of IG steps. Defaults to 30.
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        if model_id is not None:
            model_name = f'waterbirds_sweep_{model_id}'
        if os.path.exists(f'activations/{task}_{model_name}_{intervention}.p'):
            with open(f'activations/{task}_{model_name}_{intervention}.p', 'rb') as file:
                means = pickle.load(file)
        else:
            per_position = 'positional' in intervention
            means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position, device=device)
            means = means.unsqueeze(0)
            if not per_position:
                means = means.unsqueeze(0)
            with open(f'activations/{task}_{model_name}_{intervention}.p', 'wb') as file:
                pickle.dump(means, file)

    elif intervention == 'optimal':
        assert optimal_ablation_path is not None, "Path to pre-computed activations must be provided for optimal ablations"
        optimal_ablations = load_ablations(model, graph, optimal_ablation_path)
        optimal_ablations = optimal_ablations.unsqueeze(0).unsqueeze(0)

    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)
    per_example_scores = torch.zeros((len(dataloader.dataset), graph.n_forward, graph.n_backward), device='cpu', dtype=model.cfg.dtype) if get_perexample_scores else None
    labels = []
    predictions = []

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        labels.extend([this_label[0].item() if isinstance(this_label[0], torch.Tensor) else this_label[0] for this_label in label])

        clean_images = clean.to(device)
        if corrupted[0] is not None:
            corrupted_images = corrupted.to(device)

        # Here, we get our fwd / bwd hooks and the activation difference matrix
        # The forward corrupted hooks add the corrupted activations to the activation difference matrix
        # The forward clean hooks subtract the clean activations
        # The backward hooks get the gradient, and use that, plus the activation difference, for the scores
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, total_items - batch_size,
                                                                                                           batch_size,
                                                                                                           (model.cfg.image_size // model.cfg.patch_size)**2+1, scores, per_example_scores=per_example_scores)

        with torch.inference_mode():
            if intervention == 'patching':
                with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                    _ = model(corrupted_images)

            elif 'mean' in intervention:
                activation_difference += means

            elif intervention == 'optimal':
                activation_difference += optimal_ablations

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_images)
            predictions.append(clean_logits.detach().cpu())

            input_activations_clean = input_activations_corrupted - activation_difference[:, :,
                                                                    graph.forward_index(graph.nodes['input'])]

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (
                            input_activations_clean - input_activations_corrupted)
                new_input.requires_grad = True
                return new_input

            return hook_fn

        total_steps = 0
        for step in range(0, steps):
            total_steps += 1
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))],
                             bwd_hooks=bwd_hooks):
                logits = model(clean_images)
                metric_value = metric(logits, clean_logits, label)

                metric_value.backward()

    scores /= total_items
    scores /= total_steps
    if get_perexample_scores:
        per_example_scores /= total_steps
        per_example_scores = {'scores': per_example_scores, 'labels': labels, 'logits': torch.cat(predictions)}
    else:
        per_example_scores = None

    return scores, per_example_scores


def get_scores_ig_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader,
                              metric: Callable[[Tensor], Tensor], intervention: Literal[
            'patching', 'zero', 'mean', 'mean-positional', 'optimal'] = 'patching',
                              steps=30, intervention_dataloader: Optional[DataLoader] = None,
                              optimal_ablation_path: Optional[str] = None, quiet=False, device='cuda', task=None, model_name=None, get_perexample_scores=False):
    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        if os.path.exists(f'activations/{task}_{model_name}_{intervention}.p'):
            with open(f'activations/{task}_{model_name}_{intervention}.p', 'rb') as file:
                means = pickle.load(file)
        else:
            per_position = 'positional' in intervention
            means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position, device=device)
            means = means.unsqueeze(0)
            if not per_position:
                means = means.unsqueeze(0)
            with open(f'activations/{task}_{model_name}_{intervention}.p', 'wb') as file:
                pickle.dump(means, file)

    elif intervention == 'optimal':
        assert optimal_ablation_path is not None, "Path to pre-computed activations must be provided for optimal ablations"
        optimal_ablations = load_ablations(model, graph, optimal_ablation_path)
        optimal_ablations = optimal_ablations.unsqueeze(0).unsqueeze(0)

    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)
    per_example_scores = torch.zeros((len(dataloader.dataset), graph.n_forward, graph.n_backward), device='cpu', dtype=model.cfg.dtype) if get_perexample_scores else None
    labels = []
    predictions = []

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        labels.extend([this_label[0].item() if isinstance(this_label[0], torch.Tensor) else this_label[0] for this_label in label])

        clean_images = clean.to(device)
        if corrupted[0] is not None:
            corrupted_images = corrupted.to(device)

        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, total_items - batch_size, batch_size, (model.cfg.image_size // model.cfg.patch_size)**2+1, scores, per_example_scores=per_example_scores)
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(model, graph, total_items - batch_size, batch_size, (model.cfg.image_size // model.cfg.patch_size)**2+1,
                                                                                     scores, per_example_scores=per_example_scores)
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(model, graph, total_items - batch_size, batch_size, (model.cfg.image_size // model.cfg.patch_size)**2+1, scores, per_example_scores=per_example_scores)

        if intervention == 'patching':
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_images)

        elif 'mean' in intervention:
            activation_difference += means

        elif intervention == 'optimal':
            activation_difference += optimal_ablations

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_images)
            activation_difference += activations_corrupted.clone().detach() - activations_clean.clone().detach()
            predictions.append(clean_logits.detach().cpu())

        def output_interpolation_hook(k: int, clean: torch.Tensor, corrupted: torch.Tensor):
            def hook_fn(activations: torch.Tensor, hook):
                alpha = k / steps
                new_output = alpha * clean + (1 - alpha) * corrupted
                return new_output

            return hook_fn

        total_steps = 0

        nodeslist = [graph.nodes['input']]
        for layer in range(graph.cfg['n_layers']):
            nodeslist.append(graph.nodes[f'a{layer}.h0'])
            nodeslist.append(graph.nodes[f'm{layer}'])

        for node in nodeslist:
            for step in range(1, steps + 1):
                total_steps += 1

                clean_acts = activations_clean[:, :, graph.forward_index(node)]
                corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
                fwd_hooks = [(node.out_hook, output_interpolation_hook(step, clean_acts, corrupted_acts))]

                with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                    logits = model(clean_images)
                    metric_value = metric(logits, clean_logits, label)

                    metric_value.backward(retain_graph=True)

    scores /= total_items
    scores /= total_steps
    if get_perexample_scores:
        per_example_scores /= total_steps
        per_example_scores = {'scores': per_example_scores, 'labels': labels, 'logits': torch.cat(predictions)}
    else:
        per_example_scores = None

    return scores, per_example_scores


def get_scores_clean_corrupted(model: HookedTransformer, graph: Graph, dataloader: DataLoader,
                               metric: Callable[[Tensor], Tensor], quiet=False, device='cuda'):
    """Gets scores using the clean-corrupted method: like EAP-IG, but just do it on the clean and corrupted inputs, instead of all the intermediate steps.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        quiet (bool, optional): whether to silence tqdm. Defaults to False.

    Returns:
        _type_: _description_
    """

    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_images = clean.to(device)
        corrupted_images = corrupted.to(device)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, total_items - batch_size,
                                                                                                           batch_size,
                                                                                                           (model.cfg.image_size // model.cfg.patch_size)**2+1,
                                                                                                           scores)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_images)

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_images)

        total_steps = 2
        with model.hooks(bwd_hooks=bwd_hooks):
            logits = model(clean_images)
            metric_value = metric(logits, clean_logits, label)
            metric_value.backward()
            model.zero_grad()

            corrupted_logits = model(corrupted_images)
            corrupted_metric_value = metric(corrupted_logits, clean_logits, label)
            corrupted_metric_value.backward()
            model.zero_grad()

    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_information_flow_routes(model: HookedTransformer, graph: Graph, dataloader: DataLoader,
                                       quiet=False, device='cuda') -> torch.Tensor:
    """Gets scores using Ferrando et al.'s (2024) information flow routes method.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        quiet (bool, optional): whether to silence tqdm. Defaults to False.

    Returns:
        Tensor: scores based on information flow routes
    """
    # I could do some hacky overriding of make_hooks_and_matrices here but I will not
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)

    def make_hooks(n_pos: int, input_lengths: torch.Tensor) -> List[Tuple[str, Callable]]:
        output_activations = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model),
                                         device=model.cfg.device, dtype=model.cfg.dtype)

        def output_hook(index, activations, hook):
            try:
                acts = activations.detach()
                output_activations[:, :, index] = acts
            except RuntimeError as e:
                print(hook.name, output_activations[:, :, index].size(), output_activations.size())
                raise e

        # compute the score directly, without saving the input activations
        def input_hook(prev_index, bwd_index, input_lengths, activations, hook):
            acts = activations.detach()
            try:
                if acts.ndim == 3:
                    acts = acts.unsqueeze(2)
                # acts : batch pos backward hidden
                # output acts: batch pos forward hidden
                # add forward and backwards dimensions to acts and output acts respectively
                acts = acts.unsqueeze(2)
                unsqueezed_output_activations = output_activations.unsqueeze(3)

                # acts : batch pos 1 backward hidden
                # output acts: batch pos forward 1 hidden
                proximity = torch.clamp(
                    - torch.linalg.vector_norm(unsqueezed_output_activations[:, :, :prev_index] - acts, ord=1,
                                               dim=-1) + torch.linalg.vector_norm(acts, ord=1, dim=-1), min=0)
                importance = proximity / torch.sum(proximity, dim=2, keepdim=True)
                # importance: batch pos forward backward
                # aggregate over positions via sum/mean to get importance: forward backward
                # first mask out importances for padding positions
                max_len = input_lengths.max()
                mask = torch.arange(max_len, device=input_lengths.device,
                                    dtype=input_lengths.dtype).expand(len(input_lengths),
                                                                      max_len) < input_lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                # print(importance.size(), mask.size())
                importance *= mask
                importance = importance.sum(1) / input_lengths.view(-1, 1, 1)  # mean over positions
                importance = importance.sum(0)

                # importance: forward backward
                # squeezing backward dim in case it isn't real (i.e. it's an MLP)
                importance = importance.squeeze(1)
                scores[:prev_index, bwd_index] += importance

            except RuntimeError as e:
                print(hook.name, unsqueezed_output_activations[:, :, prev_index].size(), acts.size())
                raise e

        hooks = []
        node = graph.nodes['input']
        fwd_index = graph.forward_index(node)
        hooks.append((node.out_hook, partial(output_hook, fwd_index)))

        for layer in range(graph.cfg['n_layers']):
            node = graph.nodes[f'a{layer}.h0']
            fwd_index = graph.forward_index(node)
            hooks.append((node.out_hook, partial(output_hook, fwd_index)))
            prev_index = graph.prev_index(node)
            for i, letter in enumerate('qkv'):
                bwd_index = graph.backward_index(node, qkv=letter)
                hooks.append((node.qkv_inputs[i], partial(input_hook, prev_index, bwd_index, input_lengths)))

            node = graph.nodes[f'm{layer}']
            fwd_index = graph.forward_index(node)
            bwd_index = graph.backward_index(node)
            prev_index = graph.prev_index(node)
            hooks.append((node.out_hook, partial(output_hook, fwd_index)))
            hooks.append((node.in_hook, partial(input_hook, prev_index, bwd_index, input_lengths)))

        node = graph.nodes['logits']
        prev_index = graph.prev_index(node)
        bwd_index = graph.backward_index(node)
        hooks.append((node.in_hook, partial(input_hook, prev_index, bwd_index, input_lengths)))
        return hooks

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, _, _ in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_images = clean.to(device)
        input_lengths = torch.tensor([(model.cfg.image_size // model.cfg.patch_size)**2+1 for i in range(len(clean_images))]).to(device)

        hooks = make_hooks((model.cfg.image_size // model.cfg.patch_size)**2+1, input_lengths)
        with torch.inference_mode():
            with model.hooks(fwd_hooks=hooks):
                _ = model(clean_images)

    scores /= total_items

    return scores


allowed_aggregations = {'sum', 'mean'}


def attribute(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
              method: Literal[
                  'EAP', 'EAP-IG-inputs', 'clean-corrupted', 'EAP-IG-activations', 'information-flow-routes', 'exact'],
              intervention: Literal['patching', 'zero', 'mean', 'mean-positional', 'optimal'] = 'patching',
              aggregation='sum', get_perexample_scores=False,
              ig_steps: Optional[int] = None, intervention_dataloader: Optional[DataLoader] = None,
              optimal_ablation_path: Optional[str] = None, quiet=False, device='cuda', task=None, model_name=None, model_id=None):
    assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    assert model.cfg.use_split_qkv_input, "Model must be configured to use split qkv inputs (model.cfg.use_split_qkv_input)"
    assert model.cfg.use_hook_mlp_in, "Model must be configured to use hook MLP in (model.cfg.use_hook_mlp_in)"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"

    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')

    # Scores are by default summed across the d_model dimension
    # This means that scores are a [n_src_nodes, n_dst_nodes] tensor
    if method == 'EAP':
        scores = get_scores_eap(model, graph, dataloader, metric, intervention=intervention,
                                intervention_dataloader=intervention_dataloader,
                                optimal_ablation_path=optimal_ablation_path, quiet=quiet, device=device)
    elif method == 'EAP-IG-inputs':
        scores = get_scores_eap_ig(model, graph, dataloader, metric, steps=ig_steps, intervention=intervention,
                                   intervention_dataloader=intervention_dataloader, get_perexample_scores=get_perexample_scores,
                                   optimal_ablation_path=optimal_ablation_path, quiet=quiet, device=device, task=task, model_name=model_name, model_id=model_id)
    elif method == 'clean-corrupted':
        if intervention != 'patching':
            raise ValueError(f"intervention must be 'patching' for clean-corrupted, but got {intervention}")
        scores = get_scores_clean_corrupted(model, graph, dataloader, metric, quiet=quiet, device=device)
    elif method == 'EAP-IG-activations':
        scores = get_scores_ig_activations(model, graph, dataloader, metric, steps=ig_steps, intervention=intervention,
                                           intervention_dataloader=intervention_dataloader,
                                           optimal_ablation_path=optimal_ablation_path, quiet=quiet, device=device, task=task, model_name=model_name)
    elif method == 'information-flow-routes':
        scores = get_scores_information_flow_routes(model, graph, dataloader, quiet=quiet, device=device)
    elif method == 'exact':
        scores = get_scores_exact(model, graph, dataloader, metric, intervention=intervention,
                                  intervention_dataloader=intervention_dataloader,
                                  optimal_ablation_path=optimal_ablation_path, quiet=quiet, device=device)
    elif method == 'exact-optimized':
        scores = get_scores_exact_optimized(model, graph, dataloader, metric, intervention=intervention,
                                  intervention_dataloader=intervention_dataloader,
                                  optimal_ablation_path=optimal_ablation_path, quiet=quiet, device=device)
    elif method == 'exact-optimized-parallel':
        scores = get_scores_exact_optimized_parallel(model, graph, dataloader, metric, intervention=intervention,
                                  intervention_dataloader=intervention_dataloader,
                                  optimal_ablation_path=optimal_ablation_path, quiet=quiet, device=device)
    elif method == 'random':
        scores = 2 * torch.rand_like(graph.scores) - 1
    else:
        raise ValueError(
            f"method must be in ['EAP', 'EAP-IG-inputs', 'clean-corrupted', 'EAP-IG-activations', 'information-flow-routes', 'exact', 'random'], but got {method}")

    if aggregation == 'mean':
        scores /= model.cfg.d_model
    perexample_scores = None
    if isinstance(scores, tuple):
        scores, perexample_scores = scores
    graph.scores[:] = scores.to(graph.scores.device)
    if perexample_scores:
        return perexample_scores

