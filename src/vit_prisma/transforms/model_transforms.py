from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoImageProcessor
import torch

def _convert_to_rgb(image):
    return image.convert('RGB')

def get_clip_val_transforms(
    image_size=224,
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
):
    transform_list = [
        transforms.Resize(size=image_size, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
        transforms.CenterCrop(size=(image_size, image_size)),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    return transforms.Compose(transform_list)


def get_model_transforms(
    model_name
):
    print(model_name)
    if "open-clip" in model_name:
        return get_clip_val_transforms()
    else:
        try:
            img_processor = AutoImageProcessor.from_pretrained(model_name)
            img_size = img_processor.size['height']
            transform_list = [
                    transforms.Resize(size=img_size, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
                    transforms.CenterCrop(size=(img_size, img_size)),
                    _convert_to_rgb,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=img_processor.image_mean, std=img_processor.image_std)
            ]
            return transforms.Compose(transform_list)
        except:
            raise ValueError(f"Image processor for {model_name} not found. Please define the appropriate data transforms")
    
