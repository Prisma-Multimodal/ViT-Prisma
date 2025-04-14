"""
Prisma Repo
By Sonia Joseph

Copyright (c) Sonia Joseph. All rights reserved.

Inspired by TransformerLens. Some functions have been adapted from the TransformerLens project.
For more information on TransformerLens, visit: https://github.com/neelnanda-io/TransformerLens
"""

import logging
from typing import Union, Dict, Tuple, Optional, Literal
import os

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from fancy_einsum import einsum
from jaxtyping import Float

from transformers import ViTForImageClassification
from transformers import ViTConfig

from vit_prisma.configs import HookedViTConfig
from vit_prisma.models.base_transformer import HookedTransformer
from vit_prisma.models.layers.attention import Attention
from vit_prisma.models.layers.head import Head
from vit_prisma.models.layers.layer_norm import LayerNorm, LayerNormPre
from vit_prisma.models.layers.mlp import MLP
from vit_prisma.models.layers.patch_embedding import PatchEmbedding, TubeletEmbedding
from vit_prisma.models.layers.position_embedding import PosEmbedding
from vit_prisma.models.layers.transformer_block import TransformerBlock, BertBlock
from vit_prisma.prisma_tools import FactoredMatrix
from vit_prisma.prisma_tools.activation_cache import ActivationCache
from vit_prisma.prisma_tools.hook_point import HookPoint
from vit_prisma.prisma_tools.hooked_root_module import HookedRootModule

from vit_prisma.utils import devices
from vit_prisma.utils.prisma_utils import transpose
from vit_prisma.sae.sae_utils import get_deep_attr, set_deep_attr

import logging
from contextlib import contextmanager
from typing import Any, Callable, List

from vit_prisma.sae.sae import SparseAutoencoder as SAE

DTYPE_FROM_STRING = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}



class HookedViT(HookedTransformer):
    """
    Base vision model.
    Based on 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale' https://arxiv.org/abs/2010.11929.
    Adapted from TransformerLens: https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/HookedTransformer.py
    Inspiration also taken from the timm library.
    """

    def __init__(
        self,
        cfg: HookedViTConfig,
    ):
        """
        Model initialization
        """

        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedViTConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedViT object. If you want to load a "
                "pretrained model, use HookedViT.from_pretrained() instead."
            )
        self.cfg = cfg

        # ClS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.cfg.d_model))

        # Patch embeddings
        if self.cfg.is_video_transformer:
            self.embed = TubeletEmbedding(self.cfg)
        else:
            self.embed = PatchEmbedding(self.cfg)

        self.hook_embed = HookPoint()

        # Position embeddings
        self.pos_embed = PosEmbedding(self.cfg)
        self.hook_pos_embed = HookPoint()

        self.hook_full_embed = HookPoint()

        if self.cfg.layer_norm_pre:  # Put layernorm after attn/mlp layers, not before
            if self.cfg.normalization_type == "LN":
                self.ln_pre = LayerNorm(self.cfg)
            elif self.cfg.normalization_type == "LNPre":
                self.ln_pre = LayerNormPre(self.cfg)
            elif self.cfg.normalization_type is None:
                self.ln_pre = nn.Identity()
            else:
                raise ValueError(
                    f"Invalid normalization type: {self.cfg.normalization_type}"
                )
            self.hook_ln_pre = HookPoint()
        else:
            print("ln_pre not set")

        # Blocks
        if self.cfg.use_bert_block:
            block = BertBlock
        else:
            block = TransformerBlock

        self.blocks = nn.ModuleList(
            [block(self.cfg, block_index) for block_index in range(self.cfg.n_layers)]
        )
        # Final layer norm
        if self.cfg.normalization_type == "LN":
            self.ln_final = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            self.ln_final = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type is None:
            self.ln_final = nn.Identity()
        else:
            raise ValueError(
                f"Invalid normalization type: {self.cfg.normalization_type}"
            )

        self.hook_ln_final = HookPoint()

        # Final classification head
        self.head = Head(self.cfg)

        self.hook_post_head_pre_normalize = HookPoint()

        # Initialize weights
        self.init_weights()

        # Set up HookPoints
        self.setup()

    def forward(
        self,
        input: Union[Float[torch.Tensor, "batch height width channels"],],
        stop_at_layer: Optional[int] = None,
    ):
        """Forward Pass.
        Args:
            stop_at_layer Optional[int]: If not None, stop the forward pass at the specified layer.
                Exclusive - ie, stop_at_layer = 0 will only run the embedding layer, stop_at_layer =
                1 will run the embedding layer and the first transformer block, etc. Supports
                negative indexing. Useful for analysis of intermediate layers, eg finding neuron
                activations in layer 3 of a 24 layer model. Defaults to None (run the full model).
                If not None, we return the last residual stream computed.
        """

        batch_size = input.shape[0]

        embed = self.hook_embed(self.embed(input))

        if self.cfg.use_cls_token:
            cls_tokens = self.cls_token.expand(
                batch_size, -1, -1
            )  # CLS token for each item in the batch
            embed = torch.cat((cls_tokens, embed), dim=1)  # Add to embedding

        pos_embed = self.hook_pos_embed(self.pos_embed(input))


        residual = embed + pos_embed

        self.hook_full_embed(residual)

        if self.cfg.layer_norm_pre:
            residual = self.ln_pre(residual)
            residual = self.hook_ln_pre(residual)

        for block in self.blocks[:stop_at_layer]:
            residual = block(residual)
        if stop_at_layer is not None:
            return residual

        x = self.ln_final(residual)
        self.hook_ln_final(x)

        if self.cfg.classification_type == "gaap":  # GAAP

            x = x.mean(dim=1)
            print(self.cfg.return_type)
        elif self.cfg.classification_type == "cls":  # CLS token
            cls_token = x[:, 0]
            if "dino-vitb" in self.cfg.model_name:
                patches = x[:, 1:]
                patches_pooled = patches.mean(dim=1)
                x = torch.cat(
                    (cls_token.unsqueeze(-1), patches_pooled.unsqueeze(-1)), dim=-1
                )
            else:
                x = cls_token
        elif self.cfg.classification_type == "return_full_output": # return all patches
            x = x # return all patches
    

        x = x if self.cfg.return_type == "pre_logits" else self.head(x)

        self.hook_post_head_pre_normalize(x)

        if self.cfg.normalize_output:
            x = F.normalize(x, dim=-1)

        return x

    def init_weights(self):
        if self.cfg.use_cls_token:
            nn.init.normal_(self.cls_token, std=self.cfg.cls_std)
        # nn.init.trunc_normal_(self.position_embedding, std=self.cfg.pos_std)
        if self.cfg.weight_type == "he":
            for m in self.modules():
                if isinstance(m, PosEmbedding):
                    nn.init.normal_(m.W_pos, std=self.cfg.pos_std)
                elif isinstance(m, Attention):
                    nn.init.xavier_uniform_(m.W_Q)
                    nn.init.xavier_uniform_(m.W_K)
                    nn.init.xavier_uniform_(m.W_V)
                    nn.init.xavier_uniform_(m.W_O)
                elif isinstance(m, MLP):
                    nn.init.kaiming_normal_(m.W_in, nonlinearity="relu")
                    nn.init.kaiming_normal_(m.W_out, nonlinearity="relu")
                    nn.init.zeros_(m.b_out)
                    nn.init.zeros_(m.b_in)
                elif isinstance(m, Head):
                    nn.init.kaiming_normal_(m.W_H, nonlinearity="relu")
                    nn.init.zeros_(m.b_H)
                elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def run_with_cache(
        self, *model_args, return_cache_object=True, remove_batch_dim=False, **kwargs
    ) -> Tuple[
        Union[
            None,
            Float[torch.Tensor, "batch n_classes"],
        ],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """Wrapper around `run_with_cache` in HookedRootModule.

        If return_cache_object is True, this will return an ActivationCache object, with a bunch of
        useful HookedTransformer specific methods, otherwise it will return a dictionary of
        activations as in HookedRootModule.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            cache = ActivationCache(
                cache_dict, self, has_batch_dim=not remove_batch_dim
            )
            return out, cache
        else:
            return out, cache_dict

    def tokens_to_residual_directions(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes the residual directions for given labels.

        Args:
            labels (torch.Tensor): A 1D tensor of label indices with shape (batch_size,).

        Returns:
            torch.Tensor: The residual directions with shape (batch_size, d_model).
        """

        answer_residual_directions = self.head.W_H[:, labels]
        answer_residual_directions = einops.rearrange(
            answer_residual_directions, "d_model ... -> ... d_model"
        )

        return answer_residual_directions

    def fold_layer_norm(
        self, state_dict: Dict[str, torch.Tensor], fold_biases=True, center_weights=True
    ):
        """Fold Layer Norm. Can also be used to fold RMS Norm, when fold_biases and center_weights are set to False.

        Takes in a state dict from a pretrained model, formatted to be consistent with
        HookedTransformer but with LayerNorm weights and biases. Folds these into the neighbouring
        weights. See further_comments.md for more details.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of pretrained model.
            fold_biases (bool): Enables folding of LN biases. Should be disabled when RMS Norm is used.
            center_weights (bool): Enables the centering of weights after folding in LN. Should be disabled when RMS Norm is used.
        """

        # Models that use Grouped Query Attention (Only Mistral at the time of writing) prefix their K/V weights and
        # biases with an underscore in order to distinguish them, but folding the LN into them still works the same,
        # so we just add the underscore if GQA is used (i.e. if `cfg.n_key_value_heads is specified`).
        gqa = "" if self.cfg.n_key_value_heads is None else "_"

        for l in range(self.cfg.n_layers):
            # Fold ln1 into attention - it's important to fold biases first, since biases depend on
            # weights but not vice versa The various indexing is just to broadcast ln.b and ln.w
            # along every axis other than d_model. Each weight matrix right multiplies. To fold in
            # the bias, we use the W_ matrix to map it to the hidden space of the layer, so we need
            # to sum along axis -2, which is the residual stream space axis.
            if fold_biases:
                state_dict[f"blocks.{l}.attn.b_Q"] = state_dict[
                    f"blocks.{l}.attn.b_Q"
                ] + (
                    state_dict[f"blocks.{l}.attn.W_Q"]
                    * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
                ).sum(
                    -2
                )
                state_dict[f"blocks.{l}.attn.{gqa}b_K"] = state_dict[
                    f"blocks.{l}.attn.{gqa}b_K"
                ] + (
                    state_dict[f"blocks.{l}.attn.{gqa}W_K"]
                    * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
                ).sum(
                    -2
                )
                state_dict[f"blocks.{l}.attn.{gqa}b_V"] = state_dict[
                    f"blocks.{l}.attn.{gqa}b_V"
                ] + (
                    state_dict[f"blocks.{l}.attn.{gqa}W_V"]
                    * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
                ).sum(
                    -2
                )
                del state_dict[f"blocks.{l}.ln1.b"]

            state_dict[f"blocks.{l}.attn.W_Q"] = (
                state_dict[f"blocks.{l}.attn.W_Q"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            state_dict[f"blocks.{l}.attn.{gqa}W_K"] = (
                state_dict[f"blocks.{l}.attn.{gqa}W_K"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            state_dict[f"blocks.{l}.attn.{gqa}W_V"] = (
                state_dict[f"blocks.{l}.attn.{gqa}W_V"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            del state_dict[f"blocks.{l}.ln1.w"]

            # Finally, we center the weights reading from the residual stream. The output of the
            # first part of the LayerNorm is mean 0 and standard deviation 1, so the mean of any
            # input vector of the matrix doesn't matter and can be set to zero. Equivalently, the
            # output of LayerNormPre is orthogonal to the vector of all 1s (because dotting with
            # that gets the sum), so we can remove the component of the matrix parallel to this.
            if center_weights:
                state_dict[f"blocks.{l}.attn.W_Q"] -= einops.reduce(
                    state_dict[f"blocks.{l}.attn.W_Q"],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )
                state_dict[f"blocks.{l}.attn.{gqa}W_K"] -= einops.reduce(
                    state_dict[f"blocks.{l}.attn.{gqa}W_K"],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )
                state_dict[f"blocks.{l}.attn.{gqa}W_V"] -= einops.reduce(
                    state_dict[f"blocks.{l}.attn.{gqa}W_V"],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )

            # Fold ln2 into MLP
            if not self.cfg.attn_only:
                if fold_biases:
                    state_dict[f"blocks.{l}.mlp.b_in"] = state_dict[
                        f"blocks.{l}.mlp.b_in"
                    ] + (
                        state_dict[f"blocks.{l}.mlp.W_in"]
                        * state_dict[f"blocks.{l}.ln2.b"][:, None]
                    ).sum(
                        -2
                    )
                    del state_dict[f"blocks.{l}.ln2.b"]

                state_dict[f"blocks.{l}.mlp.W_in"] = (
                    state_dict[f"blocks.{l}.mlp.W_in"]
                    * state_dict[f"blocks.{l}.ln2.w"][:, None]
                )

                if self.cfg.gated_mlp:
                    state_dict[f"blocks.{l}.mlp.W_gate"] = (
                        state_dict[f"blocks.{l}.mlp.W_gate"]
                        * state_dict[f"blocks.{l}.ln2.w"][:, None]
                    )

                del state_dict[f"blocks.{l}.ln2.w"]

                if center_weights:
                    # Center the weights that read in from the LayerNormPre
                    state_dict[f"blocks.{l}.mlp.W_in"] -= einops.reduce(
                        state_dict[f"blocks.{l}.mlp.W_in"],
                        "d_model d_mlp -> 1 d_mlp",
                        "mean",
                    )

                if self.cfg.activation_name.startswith("solu"):
                    # Fold ln3 into activation
                    if fold_biases:
                        state_dict[f"blocks.{l}.mlp.b_out"] = state_dict[
                            f"blocks.{l}.mlp.b_out"
                        ] + (
                            state_dict[f"blocks.{l}.mlp.W_out"]
                            * state_dict[f"blocks.{l}.mlp.ln.b"][:, None]
                        ).sum(
                            -2
                        )

                        del state_dict[f"blocks.{l}.mlp.ln.b"]

                    state_dict[f"blocks.{l}.mlp.W_out"] = (
                        state_dict[f"blocks.{l}.mlp.W_out"]
                        * state_dict[f"blocks.{l}.mlp.ln.w"][:, None]
                    )

                    if center_weights:
                        # Center the weights that read in from the LayerNormPre
                        state_dict[f"blocks.{l}.mlp.W_out"] -= einops.reduce(
                            state_dict[f"blocks.{l}.mlp.W_out"],
                            "d_mlp d_model -> 1 d_model",
                            "mean",
                        )

                    del state_dict[f"blocks.{l}.mlp.ln.w"]

        if not self.cfg.final_rms and fold_biases:
            state_dict[f"head.b_H"] = state_dict[f"head.b_H"] + (
                state_dict[f"head.W_H"] * state_dict[f"ln_final.b"][:, None]
            ).sum(dim=-2)
            del state_dict[f"ln_final.b"]

        state_dict[f"head.W_H"] = (
            state_dict[f"head.W_H"] * state_dict[f"ln_final.w"][:, None]
        )
        del state_dict[f"ln_final.w"]

        if center_weights:
            # Center the weights that read in from the LayerNormPre
            state_dict[f"head.W_H"] -= einops.reduce(
                state_dict[f"head.W_H"], "d_model n_classes -> 1 n_classes", "mean"
            )

        print("LayerNorm folded.")

        return state_dict

    def center_writing_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Center Writing Weights.

        Centers the weights of the model that write to the residual stream - W_out, W_E, W_pos and
        W_out. This is done by subtracting the mean of the weights from the weights themselves. This
        is done in-place. See fold_layer_norm for more details.
        """
        # state_dict["embed.W_E"] = state_dict["embed.W_E"] - state_dict[
        #     "embed.W_E"
        # ].mean(-1, keepdim=True)
        if self.cfg.positional_embedding_type != "rotary":
            state_dict["pos_embed.W_pos"] = state_dict["pos_embed.W_pos"] - state_dict[
                "pos_embed.W_pos"
            ].mean(-1, keepdim=True)
        for l in range(self.cfg.n_layers):
            state_dict[f"blocks.{l}.attn.W_O"] = state_dict[
                f"blocks.{l}.attn.W_O"
            ] - state_dict[f"blocks.{l}.attn.W_O"].mean(
                -1, keepdim=True
            )  # W_O is [head_index, d_model, d_head]
            state_dict[f"blocks.{l}.attn.b_O"] = (
                state_dict[f"blocks.{l}.attn.b_O"]
                - state_dict[f"blocks.{l}.attn.b_O"].mean()
            )  # b_O is [d_model]
            if not self.cfg.attn_only:
                state_dict[f"blocks.{l}.mlp.W_out"] = state_dict[
                    f"blocks.{l}.mlp.W_out"
                ] - state_dict[f"blocks.{l}.mlp.W_out"].mean(-1, keepdim=True)
                state_dict[f"blocks.{l}.mlp.b_out"] = (
                    state_dict[f"blocks.{l}.mlp.b_out"]
                    - state_dict[f"blocks.{l}.mlp.b_out"].mean()
                )

        print("Centered weights writing to residual stream")
        return state_dict

    def fold_value_biases(self, state_dict: Dict[str, torch.Tensor]):
        """Fold the value biases into the output bias.

        Because attention patterns add up to 1, the value biases always have a constant effect on a
        head's output. Further, as the outputs of each head in a layer add together, each head's
        value bias has a constant effect on the *layer's* output, which can make it harder to
        interpret the effect of any given head, and it doesn't matter which head a bias is
        associated with. We can factor this all into a single output bias to the layer, and make it
        easier to interpret the head's output. Formally, we take b_O_new = b_O_original +
        sum_head(b_V_head @ W_O_head).
        """
        for layer in range(self.cfg.n_layers):
            # shape [head_index, d_head]
            if self.cfg.n_key_value_heads is None:
                b_V = state_dict[f"blocks.{layer}.attn.b_V"]
            else:
                b_V = state_dict[f"blocks.{layer}.attn._b_V"]
                b_V = torch.repeat_interleave(
                    b_V, dim=0, repeats=self.cfg.n_heads // self.cfg.n_key_value_heads
                )
            # [head_index, d_head, d_model]
            W_O = state_dict[f"blocks.{layer}.attn.W_O"]
            # [d_model]
            b_O_original = state_dict[f"blocks.{layer}.attn.b_O"]
            folded_b_O = b_O_original + (b_V[:, :, None] * W_O).sum([0, 1])

            state_dict[f"blocks.{layer}.attn.b_O"] = folded_b_O
            if self.cfg.n_key_value_heads is None:
                state_dict[f"blocks.{layer}.attn.b_V"] = torch.zeros_like(b_V)
            else:
                state_dict[f"blocks.{layer}.attn._b_V"] = torch.zeros_like(
                    state_dict[f"blocks.{layer}.attn._b_V"]
                )

        return state_dict

    def refactor_factored_attn_matrices(self, state_dict: Dict[str, torch.Tensor]):
        """Experimental method for managing queries, keys and values.

        As argued in [A Mathematical Framework for Transformer
        Circuits](https://transformer-circuits.pub/2021/framework/index.html), queries, keys and
        values are somewhat arbitrary intermediate terms when computing with the low rank factored
        matrices W_QK = W_Q @ W_K.T and W_OV = W_V @ W_O, and these matrices are the only thing
        determining head behaviour. But there are many ways to find a low rank factorization to a
        given matrix, and hopefully some of these are more interpretable than others! This method is
        one attempt, which makes all of the matrices have orthogonal rows or columns, W_O into a
        rotation and W_Q and W_K having the nth column in each having the same norm. The formula is
        $W_V = U @ S,W_O=Vh.T,W_Q=U@S.sqrt(),W_K=Vh@S.sqrt()$.

        More details:

        If W_OV = U @ S @ Vh.T in its singular value decomposition, (where S is in R^d_head not
        R^d_model, as W_OV is low rank), W_OV = (U @ S) @ (Vh.T) is an equivalent low rank
        factorisation, where rows/columns of each matrix are orthogonal! So setting $W_V=US$ and
        $W_O=Vh.T$ works just as well. I *think* this is a more interpretable setup, because now
        $W_O$ is just a rotation, and doesn't change the norm, so $z$ has the same norm as the
        result of the head.

        For $W_QK = W_Q @ W_K.T$ we use the refactor $W_Q = U @ S.sqrt()$ and $W_K = Vh @ S.sqrt()$,
        which is also equivalent ($S==S.sqrt() @ S.sqrt()$ as $S$ is diagonal). Here we keep the
        matrices as having the same norm, since there's not an obvious asymmetry between the keys
        and queries.

        Biases are more fiddly to deal with. For OV it's pretty easy - we just need (x @ W_V + b_V)
        @ W_O + b_O to be preserved, so we can set b_V' = 0. and b_O' = b_V @ W_O + b_O (note that
        b_V in R^{head_index x d_head} while b_O in R^{d_model}, so we need to sum b_V @ W_O along
        the head_index dimension too).

        For QK it's messy - we need to preserve the bilinear form of (x @ W_Q + b_Q) * (y @ W_K +
        b_K), which is fairly messy. To deal with the biases, we concatenate them to W_Q and W_K to
        simulate a d_model+1 dimensional input (whose final coordinate is always 1), do the SVD
        factorization on this effective matrix, then separate out into final weights and biases.
        """

        assert (
            self.cfg.positional_embedding_type != "rotary"
        ), "You can't refactor the QK circuit when using rotary embeddings (as the QK matrix depends on the position of the query and key)"

        for l in range(self.cfg.n_layers):
            # W_QK = W_Q @ W_K.T
            # Concatenate biases to make a d_model+1 input dimension
            W_Q_eff = torch.cat(
                [
                    state_dict[f"blocks.{l}.attn.W_Q"],
                    state_dict[f"blocks.{l}.attn.b_Q"][:, None, :],
                ],
                dim=1,
            )
            W_K_eff = torch.cat(
                [
                    state_dict[f"blocks.{l}.attn.W_K"],
                    state_dict[f"blocks.{l}.attn.b_K"][:, None, :],
                ],
                dim=1,
            )

            W_Q_eff_even, W_K_eff_even_T = (
                FactoredMatrix(W_Q_eff, W_K_eff.transpose(-1, -2)).make_even().pair
            )
            W_K_eff_even = W_K_eff_even_T.transpose(-1, -2)

            state_dict[f"blocks.{l}.attn.W_Q"] = W_Q_eff_even[:, :-1, :]
            state_dict[f"blocks.{l}.attn.b_Q"] = W_Q_eff_even[:, -1, :]
            state_dict[f"blocks.{l}.attn.W_K"] = W_K_eff_even[:, :-1, :]
            state_dict[f"blocks.{l}.attn.b_K"] = W_K_eff_even[:, -1, :]

            # W_OV = W_V @ W_O
            W_V = state_dict[f"blocks.{l}.attn.W_V"]
            W_O = state_dict[f"blocks.{l}.attn.W_O"]

            # Factors the bias to be consistent.
            b_V = state_dict[f"blocks.{l}.attn.b_V"]
            b_O = state_dict[f"blocks.{l}.attn.b_O"]
            effective_bias = b_O + einsum(
                "head_index d_head, head_index d_head d_model -> d_model", b_V, W_O
            )
            state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros_like(b_V)
            state_dict[f"blocks.{l}.attn.b_O"] = effective_bias

            # Helper class to efficiently deal with low rank factored matrices.
            W_OV = FactoredMatrix(W_V, W_O)
            U, S, Vh = W_OV.svd()
            state_dict[f"blocks.{l}.attn.W_V"] = U @ S.diag_embed()
            state_dict[f"blocks.{l}.attn.W_O"] = transpose(Vh)

        return state_dict

    def cuda(self):
        """Wrapper around cuda that also changes `self.cfg.device`."""
        return self.to("cuda")

    def cpu(self):
        """Wrapper around cuda that also changes `self.cfg.device`."""
        return self.to("cpu")

    def mps(self):
        """Wrapper around mps that also changes `self.cfg.device`."""
        return self.to("mps")

    def move_model_modules_to_device(self):
        device = self.cfg.device
        self.embed.proj.to(device)
        self.embed.to(devices.get_device_for_block_index(0, self.cfg))
        self.hook_embed.to(devices.get_device_for_block_index(0, self.cfg))
        if self.cfg.positional_embedding_type != "rotary":
            self.pos_embed.to(devices.get_device_for_block_index(0, self.cfg))
            self.hook_pos_embed.to(devices.get_device_for_block_index(0, self.cfg))
        if hasattr(self, "ln_final"):
            self.ln_final.to(
                devices.get_device_for_block_index(self.cfg.n_layers - 1, self.cfg)
            )
        for i, block in enumerate(self.blocks):
            block.to(devices.get_device_for_block_index(i, self.cfg))

      
    def from_local(cls, model_config: ViTConfig, checkpoint_path: str):
        model = cls(model_config)
        print(f"Loading the model locally from: {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path,
                map_location=torch.device(model_config.device),
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            return model
        else:
            raise Exception(
                "Attempting to load a Prisma ViT but no file was found at "
                f"{checkpoint_path}"
            )

    def set_use_attn_result(self, use_attn_result: bool):
        """Toggle whether to explicitly calculate and expose the result for each attention head.

        Useful for interpretability but can easily burn through GPU memory.
        """
        self.cfg.use_attn_result = use_attn_result

    def set_use_split_qkv_input(self, use_split_qkv_input: bool):
        """
        Toggles whether to allow editing of inputs to each attention head.
        """
        self.cfg.use_split_qkv_input = use_split_qkv_input

    def set_use_hook_mlp_in(self, use_hook_mlp_in: bool):
        """Toggles whether to allow storing and editing inputs to each MLP layer."""

        assert not self.cfg.attn_only, "Can't use hook_mlp_in with attn_only model"
        self.cfg.use_hook_mlp_in = use_hook_mlp_in

    def set_use_attn_in(self, use_attn_in: bool):
        """
        Toggles whether to allow editing of inputs to each attention head.
        """
        self.cfg.use_attn_in = use_attn_in

    def check_hooks_to_add(
        self,
        hook_point,
        hook_point_name,
        hook,
        dir="fwd",
        is_permanent=False,
        prepend=False,
    ) -> None:
        if hook_point_name.endswith("attn.hook_result"):
            assert (
                self.cfg.use_attn_result
            ), f"Cannot add hook {hook_point_name} if use_attn_result_hook is False"
        if hook_point_name.endswith(("hook_q_input", "hook_k_input", "hook_v_input")):
            assert (
                self.cfg.use_split_qkv_input
            ), f"Cannot add hook {hook_point_name} if use_split_qkv_input is False"
        if hook_point_name.endswith("mlp_in"):
            assert (
                self.cfg.use_hook_mlp_in
            ), f"Cannot add hook {hook_point_name} if use_hook_mlp_in is False"
        if hook_point_name.endswith("attn_in"):
            assert (
                self.cfg.use_attn_in
            ), f"Cannot add hook {hook_point_name} if use_attn_in is False"

    def accumulated_bias(
        self, layer: int, mlp_input: bool = False, include_mlp_biases=True
    ) -> Float[torch.Tensor, "layers_accumulated_over d_model"]:
        """Accumulated Bias.

        Returns the accumulated bias from all layer outputs (ie the b_Os and b_outs), up to the
        input of layer L.

        Args:
            layer (int): Layer number, in [0, n_layers]. layer==0 means no layers, layer==n_layers
                means all layers.
            mlp_input (bool): If True, we take the bias up to the input of the MLP
                of layer L (ie we include the bias from the attention output of the current layer,
                otherwise just biases from previous layers)
            include_mlp_biases (bool): Whether to include the biases of MLP layers. Often useful to
                have as False if we're expanding attn_out into individual heads, but keeping mlp_out
                as is.

        Returns:
            bias (torch.Tensor): [d_model], accumulated bias
        """

        accumulated_bias = torch.zeros(self.cfg.d_model, device=self.cls_token.device)

        for i in range(layer):
            accumulated_bias += self.blocks[i].attn.b_O
            if include_mlp_biases:
                accumulated_bias += self.blocks[i].mlp.b_out
        if mlp_input:
            assert (
                layer < self.cfg.n_layers
            ), "Cannot include attn_bias from beyond the final layer"
            accumulated_bias += self.blocks[layer].attn.b_O
        return accumulated_bias

    # Allow access to the weights via convenient properties

    @property
    def W_E(self) -> Float[torch.Tensor, "d_model channels patch_size patch_size"]:
        return self.embed.proj.weight

    @property
    def b_E(self) -> Float[torch.Tensor, "d_model"]:
        return self.embed.proj.bias

    @property
    def W_pos(self) -> Float[torch.Tensor, "n_patches+1 d_model"]:
        return self.pos_embed.W_pos

    @property
    def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        return torch.stack([block.attn.W_K for block in self.blocks], dim=0)

    @property
    def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        return torch.stack([block.attn.b_K for block in self.blocks], dim=0)

    @property
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        return torch.stack([block.attn.W_Q for block in self.blocks], dim=0)

    @property
    def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        return torch.stack([block.attn.b_Q for block in self.blocks], dim=0)

    @property
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        return torch.stack([block.attn.W_V for block in self.blocks], dim=0)

    @property
    def b_V(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        return torch.stack([block.attn.b_V for block in self.blocks], dim=0)

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        return torch.stack([block.attn.W_O for block in self.blocks], dim=0)

    @property
    def b_O(self) -> Float[torch.Tensor, "n_layers d_model"]:
        return torch.stack([block.attn.b_O for block in self.blocks], dim=0)

    @property
    def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
        return torch.stack([block.mlp.W_in for block in self.blocks], dim=0)

    @property
    def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
        return torch.stack([block.mlp.b_in for block in self.blocks], dim=0)

    @property
    def W_out(self) -> Float[torch.Tensor, "n_layers d_mlp d_model"]:
        return torch.stack([block.mlp.W_out for block in self.blocks], dim=0)

    @property
    def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
        return torch.stack([block.mlp.b_out for block in self.blocks], dim=0)

    @property
    def W_H(self) -> Float[torch.Tensor, "d_model n_classes"]:
        return self.head.W_H

    @property
    def b_H(self) -> Float[torch.Tensor, "n_classes"]:
        return self.head.b_H


class HookedSAEViT(HookedViT):

    SingleLoss = Float[torch.Tensor, ""]
    LossPerToken = Float[torch.Tensor, "batch pos-1"]
    Loss = Union[SingleLoss, LossPerToken]

    def __init__(
        self,
        *model_args: Any,
        **model_kwargs: Any,
    ):
        """Model initialization. Just HookedViT init, but adds a dictionary to keep track of attached SAEs.

        Note that if you want to load the model from pretrained weights, you should use
        :meth:`from_pretrained` instead.

        Args:
            *model_args: Positional arguments for HookedViT initialization
            **model_kwargs: Keyword arguments for HookedViT initialization
        """
        super().__init__(*model_args, **model_kwargs)
        self.acts_to_saes: Dict[str, SAE] = {}  # type: ignore

    def add_sae(self, sae: SAE, use_error_term: Optional[bool] = None):
        """Attaches an SAE to the model

        WARNING: This sae will be permanantly attached until you remove it with reset_saes. This function will also overwrite any existing SAE attached to the same hook point.

        Args:
            sae: SparseAutoencoderBase. The SAE to attach to the model
            use_error_term: (Optional[bool]) If provided, will set the use_error_term attribute of the SAE to this value. Determines whether the SAE returns input or reconstruction. Defaults to None.
        """
        act_name = sae.cfg.hook_point
        if (act_name not in self.acts_to_saes) and (act_name not in self.hook_dict):
            logging.warning(
                f"No hook found for {act_name}. Skipping. Check model.hook_dict for available hooks."
            )
            return

        if use_error_term is not None:
            if not hasattr(sae, "_original_use_error_term"):
                sae._original_use_error_term = sae.use_error_term  # type: ignore
            sae.use_error_term = use_error_term
        sae.cfg.return_out_only = True
        self.acts_to_saes[act_name] = sae
        set_deep_attr(self, act_name, sae)
        self.setup()

    def _reset_sae(self, act_name: str, prev_sae: Optional[SAE] = None):
        """Resets an SAE that was attached to the model

        By default will remove the SAE from that hook_point.
        If prev_sae is provided, will replace the current SAE with the provided one.
        This is mainly used to restore previously attached SAEs after temporarily running with different SAEs (eg with run_with_saes)

        Args:
            act_name: str. The hook_name of the SAE to reset
            prev_sae: Optional[HookedSAE]. The SAE to replace the current one with. If None, will just remove the SAE from this hook point. Defaults to None
        """
        if act_name not in self.acts_to_saes:
            logging.warning(
                f"No SAE is attached to {act_name}. There's nothing to reset."
            )
            return

        current_sae = self.acts_to_saes[act_name]
        if hasattr(current_sae, "_original_use_error_term"):
            current_sae.use_error_term = current_sae._original_use_error_term  # type: ignore
            delattr(current_sae, "_original_use_error_term")

        if prev_sae:
            set_deep_attr(self, act_name, prev_sae)
            self.acts_to_saes[act_name] = prev_sae
        else:
            set_deep_attr(self, act_name, HookPoint())
            del self.acts_to_saes[act_name]

    def reset_saes(
        self,
        act_names: Optional[Union[str, List[str]]] = None,
        prev_saes: Optional[List[Union[SAE, None]]] = None,
    ):
        """Reset the SAEs attached to the model

        If act_names are provided will just reset SAEs attached to those hooks. Otherwise will reset all SAEs attached to the model.
        Optionally can provide a list of prev_saes to reset to. This is mainly used to restore previously attached SAEs after temporarily running with different SAEs (eg with run_with_saes).

        Args:
            act_names (Optional[Union[str, List[str]]): The act_names of the SAEs to reset. If None, will reset all SAEs attached to the model. Defaults to None.
            prev_saes (Optional[List[Union[HookedSAE, None]]]): List of SAEs to replace the current ones with. If None, will just remove the SAEs. Defaults to None.
        """
        if isinstance(act_names, str):
            act_names = [act_names]
        elif act_names is None:
            act_names = list(self.acts_to_saes.keys())

        if prev_saes:
            if len(act_names) != len(prev_saes):
                raise ValueError("act_names and prev_saes must have the same length")
        else:
            prev_saes = [None] * len(act_names)  # type: ignore

        for act_name, prev_sae in zip(act_names, prev_saes):  # type: ignore
            self._reset_sae(act_name, prev_sae)

        self.setup()

    def run_with_saes(
        self,
        *model_args: Any,
        saes: Union[SAE, List[SAE]] = [],
        reset_saes_end: bool = True,
        use_error_term: Optional[bool] = None,
        **model_kwargs: Any,
    ) -> Union[
        None,
        Float[torch.Tensor, "batch pos d_vocab"],
        Loss,
        Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
    ]:
        """Wrapper around HookedViT forward pass.

        Runs the model with the given SAEs attached for one forward pass, then removes them. By default, will reset all SAEs to original state after.

        Args:
            *model_args: Positional arguments for the model forward pass
            saes: (Union[HookedSAE, List[HookedSAE]]) The SAEs to be attached for this forward pass
            reset_saes_end (bool): If True, all SAEs added during this run are removed at the end, and previously attached SAEs are restored to their original state. Default is True.
            use_error_term: (Optional[bool]) If provided, will set the use_error_term attribute of all SAEs attached during this run to this value. Defaults to None.
            **model_kwargs: Keyword arguments for the model forward pass
        """
        with self.saes(
            saes=saes, reset_saes_end=reset_saes_end, use_error_term=use_error_term
        ):
            return self(*model_args, **model_kwargs)

    def run_with_cache_with_saes(
        self,
        *model_args: Any,
        saes: Union[SAE, List[SAE]] = [],
        reset_saes_end: bool = True,
        use_error_term: Optional[bool] = None,
        return_cache_object: bool = True,
        remove_batch_dim: bool = False,
        **kwargs: Any,
    ) -> Tuple[
        Union[
            None,
            Float[torch.Tensor, "batch pos d_vocab"],
            Loss,
            Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
        ],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """Wrapper around 'run_with_cache' in HookedViT.

        Attaches given SAEs before running the model with cache and then removes them.
        By default, will reset all SAEs to original state after.

        Args:
            *model_args: Positional arguments for the model forward pass
            saes: (Union[HookedSAE, List[HookedSAE]]) The SAEs to be attached for this forward pass
            reset_saes_end: (bool) If True, all SAEs added during this run are removed at the end, and previously attached SAEs are restored to their original state. Default is True.
            use_error_term: (Optional[bool]) If provided, will set the use_error_term attribute of all SAEs attached during this run to this value. Determines whether the SAE returns input or reconstruction. Defaults to None.
            return_cache_object: (bool) if True, this will return an ActivationCache object, with a bunch of
                useful HookedViT specific methods, otherwise it will return a dictionary of
                activations as in HookedRootModule.
            remove_batch_dim: (bool) Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
            **kwargs: Keyword arguments for the model forward pass
        """
        with self.saes(
            saes=saes, reset_saes_end=reset_saes_end, use_error_term=use_error_term
        ):
            return self.run_with_cache(  # type: ignore
                *model_args,
                return_cache_object=return_cache_object,  # type: ignore
                remove_batch_dim=remove_batch_dim,
                **kwargs,
            )

    def run_with_hooks_with_saes(
        self,
        *model_args: Any,
        saes: Union[SAE, List[SAE]] = [],
        reset_saes_end: bool = True,
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],  # type: ignore
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],  # type: ignore
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        **model_kwargs: Any,
    ):
        """Wrapper around 'run_with_hooks' in HookedViT.

        Attaches the given SAEs to the model before running the model with hooks and then removes them.
        By default, will reset all SAEs to original state after.

        Args:
            *model_args: Positional arguments for the model forward pass
            act_names: (Union[HookedSAE, List[HookedSAE]]) The SAEs to be attached for this forward pass
            reset_saes_end: (bool) If True, all SAEs added during this run are removed at the end, and previously attached SAEs are restored to their original state. (default: True)
            fwd_hooks: (List[Tuple[Union[str, Callable], Callable]]) List of forward hooks to apply
            bwd_hooks: (List[Tuple[Union[str, Callable], Callable]]) List of backward hooks to apply
            reset_hooks_end: (bool) Whether to reset the hooks at the end of the forward pass (default: True)
            clear_contexts: (bool) Whether to clear the contexts at the end of the forward pass (default: False)
            **model_kwargs: Keyword arguments for the model forward pass
        """
        with self.saes(saes=saes, reset_saes_end=reset_saes_end):
            return self.run_with_hooks(
                *model_args,
                fwd_hooks=fwd_hooks,
                bwd_hooks=bwd_hooks,
                reset_hooks_end=reset_hooks_end,
                clear_contexts=clear_contexts,
                **model_kwargs,
            )

    @contextmanager
    def saes(
        self,
        saes: Union[SAE, List[SAE]] = [],
        reset_saes_end: bool = True,
        use_error_term: Optional[bool] = None,
    ):
        """
        A context manager for adding temporary SAEs to the model.
        See HookedViT.hooks for a similar context manager for hooks.
        By default will keep track of previously attached SAEs, and restore them when the context manager exits.

        Example:

        .. code-block:: python

            from transformer_lens import HookedSAETransformer, HookedSAE, HookedSAEConfig

            model = HookedSAETransformer.from_pretrained('gpt2-small')
            sae_cfg = HookedSAEConfig(...)
            sae = HookedSAE(sae_cfg)
            with model.saes(saes=[sae]):
                spliced_logits = model(text)


        Args:
            saes (Union[HookedSAE, List[HookedSAE]]): SAEs to be attached.
            reset_saes_end (bool): If True, removes all SAEs added by this context manager when the context manager exits, returning previously attached SAEs to their original state.
            use_error_term (Optional[bool]): If provided, will set the use_error_term attribute of all SAEs attached during this run to this value. Defaults to None.
        """
        act_names_to_reset = []
        prev_saes = []
        if isinstance(saes, SAE):
            saes = [saes]
        try:
            for sae in saes:
                act_names_to_reset.append(sae.cfg.hook_name)
                prev_sae = self.acts_to_saes.get(sae.cfg.hook_name, None)
                prev_saes.append(prev_sae)
                self.add_sae(sae, use_error_term=use_error_term)
            yield self
        finally:
            if reset_saes_end:
                self.reset_saes(act_names_to_reset, prev_saes)