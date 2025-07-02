import torch
import einops
from transformers import AutoModel
from vit_prisma.models.model_loader import load_hooked_model

DEVICE = "cuda"
TOLERANCE = 1e-3

def build_activation_mapping(cfg):
    """
    Return list of tuples: (hf_activation_key, ht_cache_key, reshape_fn or None)
    Matching HuggingFace activations to HookedTransformer cache keys.
    """
    mappings = []

    # Residual and norm mappings
    for l in range(cfg.n_layers):
        hf_prefix = f"encoder.layer.{l}"
        ht_prefix = f"blocks.{l}"

        mappings += [
            # Attention projections (query/key/value) outputs
            (f"{hf_prefix}.attention.query", f"{ht_prefix}.attn.hook_q", None),
            (f"{hf_prefix}.attention.key", f"{ht_prefix}.attn.hook_k", None),
            (f"{hf_prefix}.attention.value", f"{ht_prefix}.attn.hook_v", None),

            # Norm outputs
            (f"{hf_prefix}.norm1", f"{ht_prefix}.ln1.hook_normalized", None),
            (f"{hf_prefix}.norm2", f"{ht_prefix}.ln2.hook_normalized", None),

            # MLP layers
            (f"{hf_prefix}.mlp.fc1", f"{ht_prefix}.mlp.hook_pre", None),
            (f"{hf_prefix}.mlp.fc2", f"{ht_prefix}.mlp.hook_post", None),

            # Residual streams
            (f"{hf_prefix}.hook_resid_pre", f"{ht_prefix}.hook_resid_pre", None),
            (f"{hf_prefix}.hook_resid_mid", f"{ht_prefix}.hook_resid_mid", None),
            (f"{hf_prefix}.hook_resid_post", f"{ht_prefix}.hook_resid_post", None),
        ]
    return mappings


def register_hf_hooks(model, cfg):
    """
    Register forward hooks on HF model to capture needed activations in a dict,
    with keys matching the hf_activation_key in the mapping above.
    """
    activations = {}

    def save_input(key):
        def hook(module, input, output):
            # input is tuple, take first tensor
            activations[key] = input[0].detach().cpu()
        return hook

    def save_output(key):
        def hook(module, input, output):
            # If output is tuple, take the first tensor
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            activations[key] = tensor.detach().cpu()
        return hook


    for l in range(cfg.n_layers):
        layer = model.encoder.layer[l]
        prefix = f"encoder.layer.{l}"

        # Attention Q/K/V projection outputs (query/key/value are nn.Linear)
        layer.attention.query.register_forward_hook(save_output(f"{prefix}.attention.query"))
        layer.attention.key.register_forward_hook(save_output(f"{prefix}.attention.key"))
        layer.attention.value.register_forward_hook(save_output(f"{prefix}.attention.value"))

        # Norm outputs (LayerNorm)
        layer.norm1.register_forward_hook(save_output(f"{prefix}.norm1"))
        layer.norm2.register_forward_hook(save_output(f"{prefix}.norm2"))

        # MLP fc1 and fc2 outputs (Linear)
        layer.mlp.fc1.register_forward_hook(save_output(f"{prefix}.mlp.fc1"))
        layer.mlp.fc2.register_forward_hook(save_output(f"{prefix}.mlp.fc2"))

        # Residual streams: input and outputs of the whole block and attention submodule
        # resid_pre: input to entire block
        layer.register_forward_hook(save_input(f"{prefix}.hook_resid_pre"))
        # resid_mid: output of attention module
        layer.attention.register_forward_hook(save_output(f"{prefix}.hook_resid_mid"))
        # resid_post: output of entire block (layer)
        layer.register_forward_hook(save_output(f"{prefix}.hook_resid_post"))

    return activations


def compare_activations(hf_acts, ht_cache, mapping, cfg):
    for hf_key, ht_key, reshape in mapping:
        if hf_key not in hf_acts:
            print(f"Missing HF activation: {hf_key}")
            continue
        if ht_key not in ht_cache:
            print(f"Missing Hooked cache key: {ht_key}")
            continue

        hf_tensor = hf_acts[hf_key]
        ht_tensor = ht_cache[ht_key].cpu()

        # Reshape HF activations if necessary (for attn proj weights usually)
        # Here, for activations, shapes generally match, but add example if needed
        if reshape:
            try:
                hf_tensor = reshape(hf_tensor)
            except Exception as e:
                print(f"Error reshaping {hf_key} → {ht_key}: {e}")
                continue

            # For attention query/key/value activations, reshape HF to (batch, seq_len, n_heads, head_dim)
        if any(att in hf_key for att in ["attention.query", "attention.key", "attention.value"]):
            try:
                B, T, D = hf_tensor.shape
                hf_tensor = hf_tensor.reshape(B, T, cfg.n_heads, cfg.d_head)
            except Exception as e:
                print(f"Error reshaping attention proj {hf_key}: {e}")
                continue

        # Print shapes for debug
        print(f"Comparing {hf_key} (shape {hf_tensor.shape}) ↔ {ht_key} (shape {ht_tensor.shape})")

        if hf_tensor.shape != ht_tensor.shape:
            print(f"⚠️ Shape mismatch for {hf_key} vs {ht_key}: {hf_tensor.shape} != {ht_tensor.shape}")
            continue

        diff = (hf_tensor - ht_tensor).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"{hf_key} ↔ {ht_key} | max diff: {max_diff:.6e}, mean diff: {mean_diff:.6e}")

        if max_diff > TOLERANCE:
            print(f"❗ WARNING: Difference above tolerance for {hf_key}.")



def run_comparison(model_name, input_image):
    # Load HF and HookedTransformer models
    hf_model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()


    hooked_model = load_hooked_model(model_name).to(DEVICE).eval()

    print(hooked_model.cfg)

    cfg = hooked_model.cfg

    # Register HF hooks
    hf_activations = register_hf_hooks(hf_model, cfg)

    # Run HF model once to collect activations
    with torch.no_grad():
        _ = hf_model(input_image)

    # Run hooked model with cache
    with torch.no_grad():
        ht_cache = hooked_model.run_with_cache(input_image)[1]

    # Build mapping
    mapping = build_activation_mapping(cfg)

    # Compare activations
    compare_activations(hf_activations, ht_cache, mapping, cfg)


# Example usage
if __name__ == "__main__":
    input_image = torch.randn(1, 16, 3, 224, 224).to(DEVICE)
    model_name = "facebook/vjepa2-vitl-fpc64-256"
    run_comparison(model_name, input_image)
