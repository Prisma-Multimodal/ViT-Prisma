import torch
from transformers import ViTModel, AutoModel
import open_clip
from vit_prisma.models.model_loader import load_hooked_model
import HookedViT, load_hooked_model


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

input_image # some

model_name = "google/vit-base-patch16-224"