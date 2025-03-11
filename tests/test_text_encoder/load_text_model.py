import open_clip
import torch
from PIL import Image

from vit_prisma.models.base_text_transformer import HookedTextTransformer
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.utils.constants import DEVICE, BASE_DIR
from vit_prisma.utils.enums import ModelType


def get_all_layer_outputs(model, image_tensor=None, text_tensor=None):
    layer_outputs = []
    layer_names = []

    def hook_fn(module, input, output):
        layer_outputs.append(output)
        layer_names.append(type(module).__name__)

    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(image=image_tensor, text=text_tensor)

    for hook in hooks:
        hook.remove()

    return layer_outputs, layer_names


def test_loading_text_and_vision_laion_open_clip_model():
    # Given: The laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K model loaded from huggingface
    og_model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    og_model.eval()

    # Given: both a hooked text and vision model loaded via Prisma
    text_model = HookedTextTransformer.from_pretrained(
        "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K", is_timm=False, is_clip=True, model_type=ModelType.TEXT
    ).to(DEVICE)
    vision_model = HookedViT.from_pretrained(
        "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K", is_timm=False, is_clip=True
    ).to(DEVICE)
    text_model.eval()
    vision_model.eval()

    # Given: a testing image of a diagram, and some correct and incorrect labels
    image = preprocess(Image.open(BASE_DIR / "tests/test_text_encoder/test_diagram_img.png")).unsqueeze(0)
    text = tokenizer(["a diagram", "a dog", "a cat"])

    # When: the image and text is fed through both the vision and text encoder
    with torch.no_grad(), torch.cuda.amp.autocast():
        loaded_image_features, loaded_text_features, _ = og_model(image, text)
        all_outputs, layer_names = get_all_layer_outputs(og_model, image, text)

        image_features = vision_model(image)
        text_features = text_model(text)

        final_output_hooked, vision_cache = vision_model.run_with_cache(image)
        final_language_output_hooked, language_cache = text_model.run_with_cache(text)

        class_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        loaded_class_probs = (100.0 * loaded_image_features @ loaded_text_features.T).softmax(dim=-1)

    # Then: check cosine siinstitutionrities align with correct label
    assert torch.allclose(class_probs, torch.tensor([[1., 0., 0.]]), atol=0.02)
    assert torch.allclose(loaded_class_probs, torch.tensor([[1., 0., 0.]]), atol=0.02)

    # Then: check activation values the same for loaded and local models
    TOLERANCE = 1e-4
    for i, (output, name) in enumerate(zip(all_outputs, layer_names)):
        try:
            print(f"Layer {i} ({name}) output shape: {output.shape}")
        except Exception as e:
            print(f"Layer {i} ({name}) output shape: {output[0].shape}")
        if i == 0:
            output = output.flatten(2).transpose(1, 2)
            hooked_output = vision_cache['hook_embed']
            assert torch.allclose(hooked_output, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - output))}"
            print('post conv activations match')
        elif i == 1:
            hooked_output = vision_cache['hook_full_embed'] # before layer norm
            assert torch.allclose(hooked_output, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - output))}"
            print('post full embedding activations match')
        elif i == 2:
            hooked_output = vision_cache['hook_ln_pre']
            assert torch.allclose(hooked_output, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - output))}"
        elif i == 8: # First GeLU
            hooked_output = vision_cache['blocks.0.mlp.hook_post']
            assert torch.allclose(hooked_output, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - output))}"
            print("First post GeLU matches")
        elif i == 118: # Last GeLU
            hooked_output = vision_cache['blocks.11.mlp.hook_post']
            assert torch.allclose(hooked_output, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - output))}"
            print("Final post GeLU matches")
        elif i == 126:  # Text embedding
            hooked_output = language_cache['hook_embed']
            assert torch.allclose(hooked_output, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - output))}"
            print('post conv activations match')
        elif i == 132:  # First GeLU text model
            hooked_output = language_cache['blocks.0.mlp.hook_post']
            assert torch.allclose(hooked_output, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - output))}"
            print("First post GeLU matches")
        elif i == 242:  # Last GeLU text model
            hooked_output = language_cache['blocks.11.mlp.hook_post']
            assert torch.allclose(hooked_output, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - output))}"
            print("First post GeLU matches")
        elif i == 249:  # Final output
            output = output[0]
            assert torch.allclose(final_language_output_hooked, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(final_language_output_hooked - output))}"
            print("Final output matches")
        elif i > 250:
            raise Exception("Unexpected number of layers")
