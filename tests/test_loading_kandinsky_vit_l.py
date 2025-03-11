import pytest
import torch
from vit_prisma.models.base_vit import HookedViT
import numpy as np
import torch

from transformers import ViTModel


from transformers import CLIPVisionModelWithProjection

# def test_loading_kandinsky():
#     TOLERANCE = 1e-4

#     model_name = "facebook/dino-vitb16"
#     batch_size = 5
#     channels = 3
#     height = 224
#     width = 224
#     device = "cuda"
#     dtype = torch.float32



#     model = CLIPVisionModelWithProjection.from_pretrained(
#         'kandinsky-community/kandinsky-2-1-prior',
#         subfolder='image_encoder',
#         torch_dtype=dtype,
#         cache_dir = '/network/scratch/s/firstname.joseph/diffusion'
#     ).to("cuda")

#     hooked_model = HookedViT.from_pretrained('kandinsky', is_timm=False, is_clip=True, fold_ln=False, center_writing_weights=False).to(dtype) # in future, do all models
#     hooked_model.to(device)

#     with torch.random.fork_rng():
#         torch.manual_seed(1)
#         input_image = torch.rand((batch_size, channels, height, width)).to(device).to(dtype)

#     with torch.no_grad():
#         output, hooked_output = model(input_image),  hooked_model(input_image)

#         print("hooked_output", hooked_output.shape)
#         print("output", output.image_embeds.shape)

#     assert torch.allclose(hooked_output, output.image_embeds, atol=TOLERANCE)

# # main
# if __name__ == "__main__":
#     test_loading_kandinsky()

@pytest.mark.skip(reason="Reliant on files not in repo")
def test_loading_kandinsky():
    TOLERANCE = 1e-4

    batch_size = 1
    channels = 3
    height = 224
    width = 224
    device = "cuda"
    dtype = torch.float32

    def get_all_layer_outputs(model, input_tensor):
        layer_outputs = []
        layer_names = []
        
        def hook_fn(module, input, output):
            layer_outputs.append(output)
            layer_names.append(type(module).__name__)

        hooks = []
        for name, module in model.named_modules():
            hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            model(input_tensor)

        for hook in hooks:
            hook.remove()

        return layer_outputs, layer_names

    print("Loading models...")
    original_model = CLIPVisionModelWithProjection.from_pretrained(
        'kandinsky-community/kandinsky-2-1-prior',
        subfolder='image_encoder',
        torch_dtype=dtype,
        cache_dir='/network/scratch/s/firstname.joseph/diffusion'
    ).to(device)
    original_model.eval()

    hooked_model = HookedViT.from_pretrained(
        'kandinsky', 
        is_timm=False, 
        is_clip=True, 
        fold_ln=False, 
        center_writing_weights=False
    ).to(device).to(dtype)
    hooked_model.eval()

    print("Generating input...")
    torch.manual_seed(1)
    random_input = torch.rand((batch_size, channels, height, width), device=device, dtype=dtype)

    print("Getting original model outputs...")
    all_outputs, layer_names = get_all_layer_outputs(original_model, random_input)
    
    print("Getting hooked model outputs...")
    final_output_hooked, cache = hooked_model.run_with_cache(random_input)
    final_output_og = original_model(random_input).image_embeds

    print("\nCache keys available:")
    for k in cache:
        print(f"{k}: {cache[k].shape}")

    print("\nAll layer names:", layer_names)

    # Check embeddings
    try:
        embed_output = all_outputs[0]
        hooked_embed = cache['hook_embed']
        embed_output = embed_output.flatten(2).transpose(1, 2)
        assert torch.allclose(hooked_embed, embed_output, atol=TOLERANCE)
        print('Post conv activations match')
    except Exception as e:
        print(f"Error in embedding comparison: {e}")

    # Check full embeddings
    
    full_embed_output = all_outputs[1]
    hooked_full_embed = cache['hook_full_embed']
    assert torch.allclose(hooked_full_embed, full_embed_output, atol=TOLERANCE)
    print('Post full embedding activations match')


    # Check initial layer norm
    try:
        ln_pre_output = all_outputs[2]
        hooked_ln_pre = cache['hook_ln_pre']
        assert torch.allclose(hooked_ln_pre, ln_pre_output, atol=TOLERANCE)
        print('Initial layer norm matches')
    except Exception as e:
        print(f"Error in initial layer norm comparison: {e}")

    # Check first transformer block MLP
    try:
        first_gelu = cache['blocks.0.mlp.hook_post']
        corresponding_output = all_outputs[8]  # Adjust index if needed
        assert torch.allclose(first_gelu, corresponding_output, atol=TOLERANCE)
        print("First post GeLU matches")
    except Exception as e:
        print(f"Error in first GeLU comparison: {e}")

    # Check final transformer block MLP
    try:
        final_gelu = cache['blocks.11.mlp.hook_post']  # Adjust block number if needed
        corresponding_output = all_outputs[-9]  # Adjust index if needed
        assert torch.allclose(final_gelu, corresponding_output, atol=TOLERANCE)
        print("Final post GeLU matches")
    except Exception as e:
        print(f"Error in final GeLU comparison: {e}")

    # Check final layer norm
    try:
        ln_final = cache['ln_final']
        corresponding_output = all_outputs[-3]  # Adjust index if needed
        assert torch.allclose(ln_final, corresponding_output, atol=TOLERANCE)
        print("Final layer norm matches")
    except Exception as e:
        print(f"Error in final layer norm comparison: {e}")

    # Check final output
    try:
        assert torch.allclose(final_output_hooked, final_output_og, atol=TOLERANCE)
        print("Final output matches")
    except Exception as e:
        print(f"Error in final output comparison: {e}")
        print(f"Max difference: {torch.max(torch.abs(final_output_hooked - final_output_og))}")
        print(f"Hooked output shape: {final_output_hooked.shape}")
        print(f"Original output shape: {final_output_og.shape}")

    print("\nFinal output shapes:", final_output_hooked.shape, final_output_og.shape)

if __name__ == "__main__":
    test_loading_kandinsky()