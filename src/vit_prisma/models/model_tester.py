import torch
from transformers import ViTModel, AutoModel
import open_clip
from vit_prisma.models.model_loader import load_hooked_model
import einops

TOLERANCE = 1e-5
DEVICE = "cuda"


def print_divergence_info(og_output, hooked_output, model_name):
    """Print detailed divergence information between outputs."""
    diff = torch.abs(hooked_output - og_output)
    print(f"Max diff: {torch.max(diff)}")



def test_model_equivalence(model_name, input_image):
    """
    Compare outputs from original model vs HookedViT wrapper.

    Supports:
    - open-clip:* models
    - HuggingFace ViT/DINO/CLIP (AutoModel or ViTModel)
    """

    is_open_clip = model_name.startswith("open-clip:")
    is_dino = "dino" in model_name.lower()
    is_huggingface = not is_open_clip

    # === Load original model and get reference output ===
    if is_open_clip:
        # OpenCLIP model
        og_model_name = "hf-hub:" + model_name[len("open-clip:"):]
        og_model, *_ = open_clip.create_model_and_transforms(og_model_name)
        og_model.to(DEVICE)
        og_model.eval()

        with torch.no_grad():
            reference_output, *_ = og_model(input_image)

    elif is_dino:
        # DINO HuggingFace ViT
        hf_model = ViTModel.from_pretrained(model_name)
        hf_model.to(DEVICE)
        hf_model.eval()

        with torch.no_grad():
            dino_out = hf_model(input_image)
        cls_token = dino_out.last_hidden_state[:, 0]
        patches = dino_out.last_hidden_state[:, 1:]
        patches_pooled = patches.mean(dim=1)
        reference_output = torch.cat(
            (cls_token.unsqueeze(-1), patches_pooled.unsqueeze(-1)), dim=-1
        )

    elif is_huggingface:
        # Generic HuggingFace model
        hf_model = AutoModel.from_pretrained(model_name)
        hf_model.to(DEVICE)
        hf_model.eval()

        with torch.no_grad():
            out = hf_model(input_image)
        if isinstance(out, tuple):
            reference_output = out[0]
        elif hasattr(out, "last_hidden_state"):
            reference_output = out.last_hidden_state
        else:
            raise ValueError(f"Unexpected HuggingFace output format from {model_name}")

    else:
        raise ValueError(f"Unhandled model format: {model_name}")


    hooked_model = load_hooked_model(model_name)
    

    hooked_model.to(DEVICE)
    hooked_model.eval()

    if is_open_clip:
        print("hooked model config", hooked_model.cfg)

    # === Compare ===
    with torch.no_grad():
        hooked_output = hooked_model(input_image)

    print_divergence_info(reference_output, hooked_output, model_name)

    assert torch.allclose(
        reference_output, hooked_output, atol=TOLERANCE
    ), f"{model_name} output diverges! Max diff: {torch.max(torch.abs(reference_output - hooked_output))}"

def build_layerwise_mapping(cfg):
    """
    Returns a list of tuples: (hf_key, hooked_key, reshape_fn or None)
    Used for comparing layers one by one.
    """
    mappings = []

    # Embedding layer
    mappings.append(("encoder.embeddings.patch_embeddings.proj.weight", "embed.proj.weight", None))
    mappings.append(("encoder.embeddings.patch_embeddings.proj.bias", "embed.proj.bias", None))

    if getattr(cfg, "use_position_embeddings", False):
        mappings.append(("encoder.embeddings.position_embeddings", "pos_embed.W_pos", None))

    # Final LayerNorm
    mappings.append(("encoder.layernorm.weight", "ln_final.w", None))
    mappings.append(("encoder.layernorm.bias", "ln_final.b", None))

    # Transformer blocks
    for l in range(cfg.n_layers):
        hf_prefix = f"encoder.layer.{l}"
        ht_prefix = f"blocks.{l}"

        # Norms
        mappings += [
            (f"{hf_prefix}.norm1.weight", f"{ht_prefix}.ln1.w", None),
            (f"{hf_prefix}.norm1.bias", f"{ht_prefix}.ln1.b", None),
            (f"{hf_prefix}.norm2.weight", f"{ht_prefix}.ln2.w", None),
            (f"{hf_prefix}.norm2.bias", f"{ht_prefix}.ln2.b", None),
        ]

        # Attention weights and biases
        mappings += [
            (f"{hf_prefix}.attention.query.weight", f"{ht_prefix}.attn.W_Q", lambda x: einops.rearrange(x, "(h dh) d -> h d dh", h=cfg.n_heads)),
            (f"{hf_prefix}.attention.key.weight", f"{ht_prefix}.attn.W_K", lambda x: einops.rearrange(x, "(h dh) d -> h d dh", h=cfg.n_heads)),
            (f"{hf_prefix}.attention.value.weight", f"{ht_prefix}.attn.W_V", lambda x: einops.rearrange(x, "(h dh) d -> h d dh", h=cfg.n_heads)),
            (f"{hf_prefix}.attention.proj.weight", f"{ht_prefix}.attn.W_O", lambda x: einops.rearrange(x, "d (h dh) -> h dh d", h=cfg.n_heads)),

            (f"{hf_prefix}.attention.query.bias", f"{ht_prefix}.attn.b_Q", lambda x: einops.rearrange(x, "(h dh) -> h dh", h=cfg.n_heads)),
            (f"{hf_prefix}.attention.key.bias", f"{ht_prefix}.attn.b_K", lambda x: einops.rearrange(x, "(h dh) -> h dh", h=cfg.n_heads)),
            (f"{hf_prefix}.attention.value.bias", f"{ht_prefix}.attn.b_V", lambda x: einops.rearrange(x, "(h dh) -> h dh", h=cfg.n_heads)),
            (f"{hf_prefix}.attention.proj.bias", f"{ht_prefix}.attn.b_O", None),
        ]

        # MLP
        mappings += [
            (f"{hf_prefix}.mlp.fc1.weight", f"{ht_prefix}.mlp.W_in", lambda x: x.T),
            (f"{hf_prefix}.mlp.fc1.bias", f"{ht_prefix}.mlp.b_in", None),
            (f"{hf_prefix}.mlp.fc2.weight", f"{ht_prefix}.mlp.W_out", lambda x: x.T),
            (f"{hf_prefix}.mlp.fc2.bias", f"{ht_prefix}.mlp.b_out", None),
        ]

    return mappings

def compare_layerwise_dictionaries(model_name, input_image):
    import torch
    from transformers import AutoModel
    from vit_prisma.models.model_loader import load_hooked_model
    import einops

    hf_model = AutoModel.from_pretrained(model_name)
    hf_model.eval().to(DEVICE)
    hooked_model = load_hooked_model(model_name).to(DEVICE).eval()

    hf_state = hf_model.state_dict()
    hooked_state = hooked_model.state_dict()
    print("🔍 Post-load b_V first 10 entries (should NOT be zero):", hooked_state["blocks.23.attn.b_V"].flatten()[:10])

    
    print("Checking HF attention.proj.bias shape:", hf_state["encoder.layer.23.attention.proj.bias"].shape)
    print("HookedTransformer b_O shape:", hooked_model.blocks[23].attn.b_O.shape)

    print("HF value.bias first 10 entries:", hf_state["encoder.layer.23.attention.value.bias"][:10])
    print("HookedTransformer b_V first 10 entries:", hooked_model.blocks[23].attn.b_V.flatten()[:10])


    cfg = hooked_model.cfg
    

    mapping = build_layerwise_mapping(cfg)

    for hf_key, ht_key, reshape in mapping:
        if hf_key not in hf_state:
            print(f"Missing HF key: {hf_key}")
            continue
        if ht_key not in hooked_state:
            print(f"Missing Hooked key: {ht_key}")
            continue

        hf_tensor = hf_state[hf_key].detach().cpu()
        ht_tensor = hooked_state[ht_key].detach().cpu()

        # Detect if this is an attention weight needing reshape (query/key/value proj weights or output proj)
        if "attention.query.weight" in hf_key or "attention.key.weight" in hf_key or "attention.value.weight" in hf_key:
            # HF shape: (h*dh, d), HT shape: (h, d, dh)
            try:
                hf_tensor_reshaped = einops.rearrange(hf_tensor, "(h dh) d -> h d dh", h=cfg.n_heads)
            except Exception as e:
                print(f"Error reshaping {hf_key} → {ht_key}: {e}")
                continue
        elif "attention.proj.weight" in hf_key:
            # HF shape: (d, h*dh), HT shape: (h, dh, d)
            try:
                hf_tensor_reshaped = einops.rearrange(hf_tensor, "d (h dh) -> h dh d", h=cfg.n_heads)
            except Exception as e:
                print(f"Error reshaping {hf_key} → {ht_key}: {e}")
                continue
        else:
            # For everything else (biases, mlp, layernorms), just reshape if function given
            if reshape:
                try:
                    hf_tensor_reshaped = reshape(hf_tensor)
                except Exception as e:
                    print(f"Error reshaping {hf_key} → {ht_key}: {e}")
                    continue
            else:
                hf_tensor_reshaped = hf_tensor

        # Now check shapes
        if hf_tensor_reshaped.shape != ht_tensor.shape:
            print(f"Shape mismatch {hf_key} vs {ht_key}: {hf_tensor_reshaped.shape} != {ht_tensor.shape}")
            continue

        diff = torch.abs(hf_tensor_reshaped - ht_tensor)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"{hf_key} ↔ {ht_key} | max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")

    
    print("\n✅ Comparing final outputs on input_image:")
    with torch.no_grad():
        out_hf = hf_model(input_image)
        out_hooked = hooked_model(input_image)

    hf_out = out_hf.last_hidden_state if hasattr(out_hf, "last_hidden_state") else out_hf
    diff = torch.abs(hf_out - out_hooked)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"🧪 Final Output Diff — max: {max_diff:.6e}, mean: {mean_diff:.6e}")

    if not torch.allclose(hf_out, out_hooked, atol=TOLERANCE):
        print("❗ Final outputs differ beyond tolerance.")
    else:
        print("✅ Final outputs match within tolerance.")



input_image = torch.randn(1, 16, 3, 224, 224)
# input_image = input_image.permute(0, 2, 1, 3, 4)
input_image = input_image.to(DEVICE)

model_name = "facebook/vjepa2-vitl-fpc64-256"


compare_layerwise_dictionaries(model_name="facebook/vjepa2-vitl-fpc64-256", input_image=input_image)


# test_model_equivalence(model_name, input_image)

# for hf_key, ht_key, reshape_fn in build_layerwise_mapping(cfg):
#     a = hf_state_dict[hf_key].to(DEVICE)
#     b = ht_state_dict[ht_key].to(DEVICE)
#     if reshape_fn is not None:
#         a = reshape_fn(a)
#     max_diff = torch.max(torch.abs(a - b))
#     print(f"{hf_key} vs {ht_key} → max diff {max_diff.item()}") 