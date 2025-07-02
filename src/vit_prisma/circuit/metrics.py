from typing import Optional, List, Union, Literal, Tuple
from functools import partial

import pandas as pd
import torch
from torch.nn.functional import kl_div
from transformers import PreTrainedTokenizer

from vit_prisma.models.base_vit import HookedViT


def get_metric(metric_name: str,
               task: str,
               tokenizer: Optional[PreTrainedTokenizer] = None,
               model: Optional[HookedViT] = None,
               model_name: Optional[str] = None):
    """
    Get a metric function based on the metric name and task.
    The metric function basic signature is:
        metric(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False)

        TODO DOCUMENT SPECIFIC PARAMETERS AND MAKE KL/JS PARAMS MATCH THESE

    Args:
        metric_name (str): The name of the metric. Currently supported are:
            - 'kl_divergence' or 'kl' for KL Divergence between the model logits and clean logits.
            - 'js_divergence' or 'js' for JS Divergence between the model logits and clean logits.
            - 'normalized_logit' for taking the label logit and normalizing it with the highest logit.
            - 'acc' for mean accuracy.
            - 'logit_diff' or 'prob_diff' for Logit difference between
        task (str): The task of the model
        tokenizer (Optional[PreTrainedTokenizer]): The tokenizer of the model
        model (Optional[HookedTransformer]): The model
    """
    if metric_name == 'kl_divergence' or metric_name == 'kl':
        return partial(divergence, divergence_type='kl')
    elif metric_name == 'js_divergence' or metric_name == 'js':
        return partial(divergence, divergence_type='js')
    elif metric_name == 'normalized_logit':
        return normalized_logit
    elif metric_name.startswith('acc'):
        # Find the k for top-k accuracy
        if metric_name == 'acc':
            k = 1
        else:
            k = int(metric_name.split('-')[-1])
        return partial(topk_accuracy, k=k, device=model.cfg.device)
    elif metric_name == 'logit_diff' or metric_name == 'prob_diff':
        prob = (metric_name == 'prob_diff')
        if model.cfg.n_classes == 1:
            logit_diff_fn = logit_diff_binary
        else:
            logit_diff_fn = logit_diff
        return partial(logit_diff_fn, prob=prob)
    elif metric_name == 'delta_logit_diff' or metric_name == 'delta_prob_diff':
        prob = (metric_name == 'delta_prob_diff')
        return partial(delta_logit_diff, prob=prob)
    elif metric_name == "normalized_logit":
        normalized_logit_fn = normalized_logit
        return partial(normalized_logit_fn)
    elif metric_name == "sequence_logprob":
        return sequence_logprob
    else:
        raise ValueError(f"got bad metric_name: {metric_name}")


# TODO TEST THIS
def normalized_logit(circuit_logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor,
                     labels: torch.Tensor, mean: bool = True, loss: bool = False):
    circuit_logits = get_logit_positions(circuit_logits, input_length)
    circuit_outputs = circuit_logits
    labels = torch.tensor(labels, dtype=torch.long, device=circuit_outputs.device)
    good_logits = torch.gather(circuit_outputs, -1, labels.to(circuit_outputs.device))[:, 0]
    max_logits = circuit_outputs.max(dim=-1).values
    results = good_logits / max_logits
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results


# TODO TEST THIS
def topk_accuracy(logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor,
                  labels: torch.Tensor, mean: bool = True, loss: bool = False, k: int = 1, device: str = 'cuda'):
    if type(labels) == tuple:
        labels = torch.Tensor(labels).to(device)
    if labels.shape[-1] == 2:
        # If labels have both clean (good) and corrupt (bad) labels, take only the clean one
        labels = labels[:, 0]
    labels = labels.view(-1, 1)

    logits = get_logit_positions(logits, input_length)
    predictions = logits.topk(k=k, dim=-1).indices  # Get top-k predictions
    topk_acc = torch.any(predictions.eq(labels.expand_as(predictions)),
                         dim=-1).float()  # Check if any of the top-k predictions match the label
    if loss:
        topk_acc = 1 - topk_acc
    if mean:
        topk_acc = topk_acc.mean()
    return topk_acc


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor, position: int = -1) -> torch.Tensor:
    """Get the logits at a specific position of the input sequence, relative to the real (unpadded) input length; by default, get the last position

    Args:
        logits (torch.Tensor): the logits to index into
        input_length (torch.Tensor): the real (unpadded) length of each sequence
        position (int, optional): The position to get. If < 0, indexes backwards from the real unpadded input length. Defaults to -1.

    Returns:
        torch.Tensor: _description_
    """
    pos = input_length + position if position < 0 else position
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)
    logits = logits[idx, pos]
    return logits


def js_div(p: torch.tensor, q: torch.tensor):
    p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
    m = (0.5 * (p + q)).log()
    return 0.5 * (kl_div(m, p.log(), log_target=True, reduction='none').mean(-1) + kl_div(m, q.log(), log_target=True,
                                                                                          reduction='none').mean(-1))


def divergence(circuit_logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor,
               labels: torch.Tensor, divergence_type: Union[Literal['kl'], Literal['js']] = 'kl', mean=True, loss=True):
    circuit_logits = get_logit_positions(circuit_logits, input_length)
    clean_logits = get_logit_positions(clean_logits, input_length)

    circuit_probs = torch.softmax(circuit_logits, dim=-1)
    clean_probs = torch.softmax(clean_logits, dim=-1)

    if divergence_type == 'kl':
        results = kl_div(circuit_probs.log(), clean_probs.log(), log_target=True, reduction='none').mean(-1)
    elif divergence_type == 'js':
        results = js_div(circuit_probs, clean_probs)
    else:
        raise ValueError(f"Expected divergence_type of 'kl' or 'js', but got '{divergence_type}'")
    return results.mean() if mean else results


def logit_diff_binary(circuit_logits: torch.Tensor, clean_logits: torch.Tensor, labels: torch.Tensor, mean=True,
                      prob=False, loss=False):
    circuit_logits = torch.cat([-circuit_logits, circuit_logits], dim=-1)
    circuit_outputs = torch.sigmoid(circuit_logits) if prob else circuit_logits
    # good_bad = torch.gather(circuit_outputs, -1, labels.to(circuit_outputs.device))
    labels = torch.tensor(labels, dtype=torch.long, device=circuit_outputs.device)
    good_bad = torch.gather(circuit_outputs, -1, labels)
    results = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        # remember it's reversed to make it a loss
        results = -results
    if mean:
        results = results.mean()
    return results


def delta_logit_diff(circuit_logits: torch.Tensor, clean_logits: torch.Tensor, labels: torch.Tensor, mean=True,
                     prob=False, loss=False):
    circuit_outputs = torch.softmax(circuit_logits, dim=-1) if prob else circuit_logits
    clean_outputs = torch.softmax(clean_logits, dim=-1) if prob else clean_logits
    # good_bad = torch.gather(circuit_outputs, -1, labels.to(circuit_outputs.device))
    labels = torch.tensor(labels, dtype=torch.long, device=circuit_outputs.device)
    good_bad = torch.gather(circuit_outputs, -1, labels)
    clean_good_bad = torch.gather(clean_outputs, -1, labels)
    results = good_bad[:, 0] - good_bad[:, 1]
    clean_results = clean_good_bad[:, 0] - clean_good_bad[:, 1]
    if loss:
        # remember it's reversed to make it a loss
        results = -(results - clean_results)
    if mean:
        results = (results - clean_results).mean()
    return results


def logit_diff(circuit_logits: torch.Tensor, clean_logits: torch.Tensor, labels: torch.Tensor, mean=True, prob=False,
               loss=False):
    circuit_outputs = torch.softmax(circuit_logits, dim=-1) if prob else circuit_logits
    # good_bad = torch.gather(circuit_outputs, -1, labels.to(circuit_outputs.device))
    labels = torch.tensor(labels, dtype=torch.long, device=circuit_outputs.device)
    good_bad = torch.gather(circuit_outputs, -1, labels)
    results = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        # remember it's reversed to make it a loss
        results = -results
    if mean:
        results = results.mean()
    return results


gt_indices_dict = {}


def get_year_indices(model_name: str, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    if model_name not in gt_indices_dict:
        if 'gpt2' in model_name:
            # GPT-2 tokenizes 4-digit years into groups of 2 and 2; we care about the latter 2
            gt_indices_dict[model_name] = torch.tensor(
                [tokenizer(f'{year:02d}', add_special_tokens=False).input_ids[0] for year in range(100)])
        elif 'Llama-3' in model_name:
            # Llama-3 tokenizes 4-digit years into groups of 3 and 1; we care about both
            three_digit_indices = torch.tensor(
                [tokenizer(f'{digit:03d}', add_special_tokens=False).input_ids[0] for digit in range(1000)])
            single_digit_indices = torch.tensor(
                [tokenizer(f'{digit}', add_special_tokens=False).input_ids[-1] for digit in range(10)])
            gt_indices_dict[model_name] = (three_digit_indices, single_digit_indices)
        elif 'gemma-2' in model_name or 'Qwen' in model_name:
            # Gemma-2 and Qwen tokenize 4 digit years into 4 individual digits
            gt_indices_dict[model_name] = torch.tensor(
                [tokenizer(f'{digit}', add_special_tokens=False).input_ids[-1] for digit in range(10)])
        else:
            raise ValueError(f"Model name '{model_name}' not recognized")
    return gt_indices_dict[model_name]


def logit_diff_greater_than(circuit_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor,
                            labels: torch.Tensor, model_name: str,
                            indices: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], mean: bool = True,
                            prob: bool = False, loss: bool = False):
    """Computes greater-than, which may involve multiple tokens on non-GPT-2 models. Note that we only care about the last two digits, even if the model
    doesn't tokenize the start / end year into XXYY, so we may have to account for this.

    Args:
        circuit_logits (torch.Tensor): The circuit logits
        corrupted_logits (torch.Tensor): (unused)
        input_length (torch.Tensor): _description_
        labels (torch.Tensor): _description_
        model_name (str): _description_
        indices (torch.Tensor): The indices of the relevant years (may be either a torch.Tensor or Tuple thereof)
        mean (bool, optional): _description_. Defaults to True.
        prob (bool, optional): _description_. Defaults to False.
        loss (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if not prob:
        raise ValueError("Can't compute multi-token logit diff")

    probs = torch.softmax(circuit_logits, dim=-1)

    results = []
    if 'gpt2' in model_name:
        # GPT-2 tokenizes 4-digit years into groups of 2 and 2; we care about the latter 2
        probs_yr = get_logit_positions(probs, input_length, position=-2)
        year_indices = indices
        for year_probs, (year,) in zip(probs_yr[:, year_indices], labels):
            good_year_probs = year_probs[year + 1:]
            bad_year_probs = year_probs[:year + 1]

            good_probs = good_year_probs.sum()
            bad_probs = bad_year_probs.sum()
            results.append(good_probs - bad_probs)

    elif 'Llama-3' in model_name:
        # Llama-3 tokenizes 4-digit years into groups of 3 and 1; we care about both
        # first is indices for the three-digit millenium, century, and decade. second is for the single digit year
        probs_dec = get_logit_positions(probs, input_length, position=-3)
        probs_yr = get_logit_positions(probs, input_length, position=-2)
        decade_indices, digit_indices = indices
        for decade_probs, year_probs, (decade, year) in zip(probs_dec[:, decade_indices], probs_yr[:, digit_indices],
                                                            labels):
            decade_prob = decade_probs[decade]
            good_decade_probs = decade_probs[decade + 1:]
            bad_decade_probs = decade_probs[:decade]

            good_year_probs = year_probs[year + 1:]
            bad_year_probs = year_probs[:year + 1]

            good_probs = good_decade_probs.sum() + decade_prob * good_year_probs.sum()
            bad_probs = bad_decade_probs.sum() + decade_prob * bad_year_probs.sum()
            results.append(good_probs - bad_probs)

    elif 'gemma-2' in model_name or 'Qwen' in model_name:
        # Gemma-2 and Qwen tokenize 4 digit years into 4 individual digits
        probs_dec = get_logit_positions(probs, input_length, position=-3)
        probs_yr = get_logit_positions(probs, input_length, position=-2)
        digit_indices = indices
        for decade_probs, year_probs, (decade, year) in zip(probs_dec[:, digit_indices], probs_yr[:, digit_indices],
                                                            labels):
            decade_prob = decade_probs[decade]
            good_decade_probs = decade_probs[decade + 1:]
            bad_decade_probs = decade_probs[:decade]

            good_year_probs = year_probs[year + 1:]
            bad_year_probs = year_probs[:year + 1]

            good_probs = good_decade_probs.sum() + decade_prob * good_year_probs.sum()
            bad_probs = bad_decade_probs.sum() + decade_prob * bad_year_probs.sum()
            results.append(good_probs - bad_probs)
    else:
        raise ValueError(f"Model name '{model_name}' not recognized")

    results = torch.stack(results)
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results


def _batch_index_matrix(M, a):
    """
    Efficiently index a 3D tensor M (B x L x |V|) using batched indices from a

    Args:
        M: Tensor of shape (B, L, V)
        a: Tensor of shape (B, K) containing indices into the V dimension for each batch

    Returns:
        Tensor of shape (B, K) containing the indexed values
    """
    B, L, V = M.shape
    _, K = a.shape

    # Create indices for the batch dimension
    batch_idx = torch.arange(B).unsqueeze(1).expand(B, K)

    # Create indices for the L dimension
    L_idx = torch.arange(K).unsqueeze(0).expand(B, K)

    # Index the matrix using advanced indexing
    result = M[batch_idx, L_idx, a]

    return result


def sequence_logprob(circuit_logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor,
                     labels: torch.Tensor, mean=True, loss=False):
    circuit_outputs = torch.nn.functional.log_softmax(circuit_logits, dim=-1)
    print("circuit outputs shape:", circuit_outputs.shape)
    # good_bad = torch.gather(circuit_outputs, -1, labels.to(circuit_outputs.device))
    padded_labels = []
    max_label_len = 0
    for labelset in labels:
        max_label_len = max(max_label_len, len(labelset[0]))
    for labelset in labels:
        padded_labels.append(labelset)
        if len(labelset[0]) < max_label_len:
            for i in range(max_label_len - len(labelset[0])):
                padded_labels[-1][0].append(0)
                padded_labels[-1][1].append(0)

    labels = torch.tensor(padded_labels, dtype=torch.long, device=circuit_outputs.device)[:, 0, :]
    circuit_logits_targetseq = circuit_outputs[:, -labels.shape[-1]:, :]
    print("labels shape:", labels.shape)
    logprobs = _batch_index_matrix(circuit_logits, labels)
    seq_logprob = logprobs.sum(dim=-1)  # one sequence logprob per element in batch
    print("fn output shape:", seq_logprob.shape)
    if loss:
        # remember it's reversed to make it a loss
        logprobs = -logprobs
    if mean:
        logprobs = logprobs.mean()
    return logprobs