import timm
import torch

from vit_prisma.models.base_vit import HookedViT


#currently only vit_base_patch16_224 supported (config loading issue)
def test_loading_timm():
    TOLERANCE = 1e-4

    model_name = "vit_base_patch16_224"
    batch_size = 5
    channels = 3
    height = 224
    width = 224
    device = "cpu"

    hooked_model = HookedViT.from_pretrained(model_name)
    hooked_model.to(device)
    timm_model = timm.create_model(model_name, pretrained=True)
    timm_model.to(device)

    with torch.random.fork_rng():
        torch.manual_seed(1)
        input_image = torch.rand((batch_size, channels, height, width)).to(device)

    hooked_output, timm_output = hooked_model(input_image), timm_model(input_image)

    assert torch.allclose(hooked_output, timm_output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - timm_output))}"


test_loading_timm()
