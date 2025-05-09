from vit_prisma.sae.sae import SparseAutoencoder
import torch.nn as nn
import torch
import einops

class Transcoder(SparseAutoencoder):

    def initialize_sae_weights(self):

        self.W_skip = nn.Parameter(self.initialize_weights(self.d_in, self.d_in)) if self.cfg.transcoder_with_skip_connection else None
        
        self.W_dec = nn.Parameter(self.initialize_weights(self.d_sae, self.cfg.d_out))

        self.W_enc = nn.Parameter(self.initialize_weights(self.d_in, self.d_sae))

        self.b_enc = nn.Parameter(
            torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
        )

        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=self.dtype, device=self.device)
        )

        self.b_dec_out = nn.Parameter(
            torch.zeros(self.cfg.d_out, dtype=self.dtype, device=self.device)
        )


    def encode(self, x: torch.Tensor, return_hidden_pre: bool = False):
        # move x to correct dtype
        x = x.to(self.dtype)

        sae_in = self.run_time_activation_norm_fn_in(x)

        sae_in = self.hook_sae_in(
            sae_in - self.b_dec
        )  

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(
                sae_in,
                self.W_enc,
                "... d_in, d_in d_sae -> ... d_sae",
            )
            + self.b_enc
        )

        feature_acts = self.hook_hidden_post(self.activation_fn(hidden_pre))

        if return_hidden_pre:
            return sae_in, feature_acts, hidden_pre

        return sae_in, feature_acts

    def decode(self, features: torch.Tensor):
        sae_out = self.hook_sae_out(
            einops.einsum(
                features,
                self.W_dec,
                "... d_sae, d_sae d_out -> ... d_out",
            )
            + self.b_dec_out
        )

        return sae_out

    def forward(
        self, x: torch.Tensor, y: torch.Tensor = None, dead_neuron_mask: torch.Tensor = None, *args, **kwargs
    ):

        _, feature_acts, hidden_pre = self.encode(x, return_hidden_pre=True)
        sae_out = self.decode(feature_acts)

        if self.W_skip is not None:
            # sae_out += x.to(self.dtype) @ self.W_skip.mT
            identity = torch.eye(self.d_in, dtype=self.dtype, device=self.device)
            sae_out += x.to(self.dtype) @ (identity + self.W_skip.mT)


        sae_out = self.run_time_activation_norm_fn_out(sae_out)

        if y is not None:
            mse_loss = self._compute_mse_loss(y, sae_out)
        else:
            mse_loss = 0

        # Compute ghost residual loss if required
        if self.cfg.use_ghost_grads and self.training and dead_neuron_mask is not None:
            mse_loss_ghost_resid = self._compute_ghost_residual_loss(
                x, sae_out, hidden_pre, dead_neuron_mask
            )
        else:
            mse_loss_ghost_resid = self.zero_loss

        # Compute sparsity loss
        sparsity = feature_acts.norm(p=self.lp_norm, dim=1).mean(dim=(0,))

        # Combine losses based on activation function type
        l1_loss = (
            self.l1_coefficient * sparsity
            if self.cfg.activation_fn_str != "topk"
            else None
        )

        loss = mse_loss + (l1_loss if l1_loss is not None else 0) + mse_loss_ghost_resid

        # Placeholder for auxiliary reconstruction loss
        aux_reconstruction_loss = torch.tensor(0.0) 

        if hasattr(self.cfg, "return_out_only"): # to work with HookedSAEViT efficiently
            if self.cfg.return_out_only:    
                return sae_out

        return (
            sae_out,
            feature_acts,
            loss,
            mse_loss,
            l1_loss,
            mse_loss_ghost_resid,
            aux_reconstruction_loss,
        )