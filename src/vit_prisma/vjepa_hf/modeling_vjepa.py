from typing import Dict, List, Optional, Set, Tuple, Union, Any

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput, logging
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import to_numpy_array

import PIL
from PIL import Image

import cv2
from torchvision.transforms import functional as tvf

import os
import numpy as np
import math
import numbers


logger = logging.get_logger(__name__)

## Utility functions


class Compose(object):
    """Composes several transforms
    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def resize_clip(clip, size, interpolation="bilinear"):
    if isinstance(clip[0], np.ndarray) or isinstance(clip[0], torch.Tensor):
        if isinstance(size, numbers.Number):
            if clip[0].shape[-1] == 3:
                im_h, im_w, im_c = clip[0].shape
            else:
                assert clip[0].shape[0] == 3
                im_c, im_h, im_w = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[0], size[1]

        if isinstance(clip[0], np.ndarray):
            if interpolation == "bilinear":
                np_inter = cv2.INTER_LINEAR
            else:
                np_inter = cv2.INTER_NEAREST
            scaled = [cv2.resize(img, size, interpolation=np_inter) for img in clip]
        else:  # isinstance(clip[0], torch.Tensor)
            if interpolation == "bilinear":
                np_inter = tvf.InterpolationMode.BILINEAR
            else:
                np_inter = tvf.InterpolationMode.NEAREST
            size = (size[1], size[0])  # torchvision transformers expect the size in (h, w) order.
            scaled = [tvf.resize(img, size, interpolation=np_inter) for img in clip]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == "bilinear":
            pil_inter = PIL.Image.BILINEAR
        else:
            pil_inter = PIL.Image.NEAREST
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError(
            "Expected numpy.ndarray or PIL.Image or torch.Tensor" + "but got list of {0}".format(type(clip[0]))
        )
    return scaled


class Resize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, size, interpolation="nearest"):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        resized = resize_clip(clip, self.size, interpolation=self.interpolation)
        return resized


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray) or isinstance(clip[0], torch.Tensor):
        if clip[0].shape[-1] == 3:
            cropped = [img[min_h : min_h + h, min_w : min_w + w, :] for img in clip]
        else:
            assert clip[0].shape[0] == 3
            cropped = [img[:, min_h : min_h + h, min_w : min_w + w] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip]

    else:
        raise TypeError(
            "Expected numpy.ndarray or PIL.Image or torch.Tensor):" + "but got list of {0}".format(type(clip[0]))
        )
    return cropped


class CenterCrop(object):
    """Extract center crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray) or isinstance(clip[0], torch.Tensor):
            if clip[0].shape[-1] == 3:
                im_h, im_w, im_c = clip[0].shape
            else:
                assert clip[0].shape[0] == 3
                im_c, im_h, im_w = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image or torch.Tensor" + "but got list of {0}".format(type(clip[0]))
            )
        if w > im_w or h > im_h:
            error_msg = (
                "Initial image size should be larger then "
                "cropped size but got cropped sizes : ({w}, {h}) while "
                "initial image is ({im_w}, {im_h})".format(im_w=im_w, im_h=im_h, w=w, h=h)
            )
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.0))
        y1 = int(round((im_h - h) / 2.0))
        cropped = crop_clip(clip, y1, x1, h, w)

        return cropped


def convert_img(img):
    """Converts (H, W, C) numpy.ndarray to (C, W, H) format"""
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    return img


class ClipToTensor(object):
    """Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0]
    """

    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.channel_nb = channel_nb
        self.div_255 = div_255
        self.numpy = numpy

    def __call__(self, clip):
        """
        Args: clip (list of numpy.ndarray): clip (list of images)
        to be converted to tensor.
        """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, "Got {0} instead of 3 channels".format(ch)
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
        elif isinstance(clip[0], torch.Tensor):
            tensor_clip = torch.stack(clip)
            # Converting (T, C, H, W) -> (C, T, H, W) to match what `convert_img` followed by
            # `np_clip[:, img_idx, :, :] = img` does for other data types.
            tensor_clip = tensor_clip.permute(1, 0, 2, 3)
            if not isinstance(tensor_clip, torch.FloatTensor):
                tensor_clip = tensor_clip.float()
            if self.div_255:
                tensor_clip = torch.div(tensor_clip, 255)
            return tensor_clip
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image or torch.Tensor\
            but got list of {0}".format(
                    type(clip[0])
                )
            )

        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])

        # Convert
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError(
                    "Expected numpy.ndarray or PIL.Image\
                but got list of {0}".format(
                        type(clip[0])
                    )
                )
            img = convert_img(img)
            np_clip[:, img_idx, :, :] = img

        if self.numpy:
            if self.div_255:
                np_clip = np_clip / 255.0
            return np_clip

        else:
            tensor_clip = torch.from_numpy(np_clip)

            if not isinstance(tensor_clip, torch.FloatTensor):
                tensor_clip = tensor_clip.float()
            if self.div_255:
                tensor_clip = torch.div(tensor_clip, 255)
            return tensor_clip


def _is_tensor_clip(clip):
    return torch.is_tensor(clip) and clip.ndimension() == 4


def fn_normalize(clip, mean, std, inplace=False):
    if not _is_tensor_clip(clip):
        raise TypeError("tensor is not a torch clip.")

    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

    return clip


class Normalize(object):
    """Normalize a clip with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        """
        Args:
            clip (Tensor): Tensor clip of size (T, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor clip.
        """
        return fn_normalize(clip, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False):
    """
    grid_size: int of the grid height and width
    grid_depth: int of the grid depth
    returns:
        pos_embed: [grid_depth*grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_depth*grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(
        grid_h, grid_d, grid_w
    )  # order of meshgrid is very important for indexing as [d,h,w]

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    returns:
        pos_embed: [grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # order of meshgrid is very important for indexing as [h, w]

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size, embed_dim] (w/ cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


## Image processor (transforms)


class VJEPAImageProcessor:
    def __init__(self, crop_size: int = 224, normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):

        short_side_size = int(crop_size * 256 / 224)
        self.transform = Compose(
            [
                Resize(short_side_size, interpolation="bilinear"),
                CenterCrop(size=(crop_size, crop_size)),
                ClipToTensor(),
                Normalize(mean=normalize[0], std=normalize[1]),
            ]
        )
        self.size = [crop_size]
        self.crop_size = {"height": crop_size, "width": crop_size}

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = to_numpy_array(images)  # H x W x 3
            assert images.ndim == 3
            images = np.expand_dims(images, axis=0)  # T=1 x H x W x C
        else:
            # to adapt video data
            images = np.concatenate(
                [np.expand_dims(to_numpy_array(image), axis=0) for image in images], axis=0
            )  # T x H x W x C

        images = self.transform(images)  # C x T x H x W, where T=1 for image
        images = images.permute(1, 0, 2, 3).unsqueeze(0).numpy()  # add batch, B x T x C x H x W
        images = list(images)
        data = {"pixel_values": images}  # list of T x C x H x W

        return BatchFeature(data=data, tensor_type=return_tensors)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.preprocess(*args, **kwds)


## VJEPA config


class VJEPAConfig(PretrainedConfig):
    model_type = "vjepa_vision_model"

    def __init__(
        self,
        model_name="vit_large",
        patch_size=16,
        crop_size=224,
        frames_per_clip=16,
        tubelet_size=2,
        use_sdpa=False,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        hidden_size=-1,
        in_chans=3,
        num_attention_heads=12,
        num_hidden_layers=12,
        drop_path_rate=0.0,
        mlp_ratio=4.0,
        is_causal=False,
        layer_norm_eps=1e-6,
        qkv_bias=True,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        hidden_act="gelu",
        initializer_range=0.02,
        use_rope=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_name = model_name
        self.crop_size = crop_size
        self.img_height = crop_size
        self.img_width = crop_size
        self.frames_per_clip = frames_per_clip
        self.patch_size = patch_size
        self.num_frames = frames_per_clip
        self.tubelet_size = tubelet_size
        self.uniform_power = uniform_power
        self.use_sdpa = use_sdpa
        self.use_SiLU = use_SiLU
        self.wide_SiLU = wide_SiLU
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.drop_path_rate = drop_path_rate
        self.mlp_ratio = mlp_ratio
        self.is_causal = is_causal
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.image_size = crop_size
        self.use_rope = use_rope

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from SigLipConfig
        if config_dict.get("model_type") == "vjepa":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


## Modules


class VJEPAPatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, config: VJEPAConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_patches = (config.crop_size // self.patch_size) * (config.crop_size // self.patch_size)
        self.proj = nn.Conv2d(
            config.in_chans, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VJEPAPatchEmbeddings3D(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        config: VJEPAConfig,
    ):
        super().__init__()
        self.patch_size = config.patch_size
        self.tubelet_size = config.tubelet_size

        self.num_patches = (
            (config.frames_per_clip // config.tubelet_size)
            * (config.crop_size // config.patch_size)
            * (config.crop_size // config.patch_size)
        )

        self.proj = nn.Conv3d(
            in_channels=config.in_chans,
            out_channels=config.hidden_size,
            kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
            stride=(config.tubelet_size, config.patch_size, config.patch_size),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VJEPAEmbeddings(nn.Module):
    """
    Construct mask token, position and patch embeddings.
    """

    def __init__(self, config: VJEPAConfig) -> None:
        super().__init__()

        # self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.config = config
        self.is_video = config.frames_per_clip > 1
        # self.mask_token = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.patch_embeddings = VJEPAPatchEmbeddings3D(config) if self.is_video else VJEPAPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        # position embeddings
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size), requires_grad=False)
        self._init_pos_embed(self.position_embeddings.data)  # sincos pos-embed

        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def _init_pos_embed(self, pos_embed):
        grid_size = (
            self.config.crop_size // self.config.patch_size
        )  # TODO: update; initialization currently assumes square input
        if self.is_video:
            grid_depth = self.config.frames_per_clip // self.config.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                self.config.hidden_size,
                grid_size,
                grid_depth,
                cls_token=False,
                uniform_power=self.config.uniform_power,
            )
        else:
            sincos = get_2d_sincos_pos_embed(self.config.hidden_size, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def interpolate_pos_encoding(self, x, pos_embed):

        _, N, dim = pos_embed.shape

        if self.is_video:

            # If pos_embed already corret size, just return
            _, _, T, H, W = x.shape
            if H == self.config.img_height and W == self.config.img_width and T == self.config.frames_per_clip:
                return pos_embed

            # Convert depth, height, width of input to be measured in patches
            # instead of pixels/frames
            T = T // self.config.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size

            # Compute the initialized shape of the positional embedding measured
            # in patches
            N_t = self.config.frames_per_clip // self.config.tubelet_size
            N_h = self.config.img_height // self.patch_size
            N_w = self.config.img_width // self.patch_size
            assert N_h * N_w * N_t == N, "Positional embedding initialized incorrectly"

            # Compute scale factor for spatio-temporal interpolation
            scale_factor = (T / N_t, H / N_h, W / N_w)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode="trilinear",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed

        else:

            # If pos_embed already corret size, just return
            _, _, H, W = x.shape
            if H == self.config.img_height and W == self.config.img_width:
                return pos_embed

            # Compute scale factor for spatial interpolation
            npatch = (H // self.patch_size) * (W // self.patch_size)
            scale_factor = math.sqrt(npatch / N)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode="bicubic",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return pos_embed

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        if len(pixel_values.shape) != 5:
            pixel_values = pixel_values.unsqueeze(2)  # add a new dimension for time
            pixel_values = pixel_values.repeat(1, 1, self.config.frames_per_clip, 1, 1) 
        batch_size, c, t, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.proj.weight.dtype

        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        # if bool_masked_pos is not None:
        #     embeddings = torch.where(bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings)

        # add positional encoding to each token
        # ignore if we are using Rope
        if not self.config.use_rope:
            embeddings = embeddings + self.interpolate_pos_encoding(pixel_values, self.position_embeddings)

        # embeddings = self.dropout(embeddings)

        return embeddings


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->Dinov2->VJEPA
class VJEPASelfAttention(nn.Module):
    def __init__(self, config: VJEPAConfig) -> None:
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # Note: this is the self.scale in our applications
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = self.proj(context_layer.view(new_context_layer_shape))

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


def rotate_queries_or_keys(x, pos):
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    # siinstitutionr to inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    # they are computing this every time. instead HF style is to compute the inv_freq once and store it
    # -- compute angle for each position
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = torch.einsum("..., f -> ... f", pos, omega)  # (..., N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)

    emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)

    # --
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(
        dim=-1,
    )
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


class VJEPARopeSelfAttention(nn.Module):
    def __init__(self, config: VJEPAConfig) -> None:
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.grid_size = self.config.crop_size // self.config.patch_size
        self.grid_depth = self.config.frames_per_clip // self.config.tubelet_size

        self.d_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.h_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.w_dim = int(2 * ((self.attention_head_size // 3) // 2))

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _get_frame_pos(self, ids):
        tokens_per_frame = int(self.grid_size * self.grid_size)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids):
        # Remove frame component from ids
        tokens_per_frame = int(self.grid_size * self.grid_size)
        frame_ids = self._get_frame_pos(ids)
        ids = ids - tokens_per_frame * frame_ids
        # --
        tokens_per_row = self.grid_size
        return ids // tokens_per_row

    def get_position_ids(self, device):
        ids = torch.arange(int(self.grid_depth * self.grid_size * self.grid_size), device=device)
        tokens_per_frame = int(self.grid_size * self.grid_size)
        frame_ids = self._get_frame_pos(ids)
        # --
        tokens_per_row = self.grid_size
        height_ids = self._get_height_pos(ids)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def apply_rotary_embeddings(self, qk, pos_ids):
        d_mask, h_mask, w_mask = pos_ids
        s = 0
        qkd = rotate_queries_or_keys(qk[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        qkh = rotate_queries_or_keys(qk[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        qkw = rotate_queries_or_keys(qk[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim
        # Combine rotated dimension
        if s < self.attention_head_size:
            qkr = qk[..., s:]
            qk = torch.cat([qkd, qkh, qkw, qkr], dim=-1)
        else:
            qk = torch.cat([qkd, qkh, qkw], dim=-1)
        return qk

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        pos_ids = self.get_position_ids(hidden_states.device)
        key_layer = self.apply_rotary_embeddings(key_layer, pos_ids)
        query_layer = self.apply_rotary_embeddings(query_layer, pos_ids)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # Note: this is the self.scale in our applications
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = self.proj(context_layer.view(new_context_layer_shape))

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class VJEPASdpaSelfAttention(VJEPASelfAttention):
    def __init__(self, config: VJEPAConfig) -> None:
        super().__init__(config)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "VJEPAModel is using VJEPASdpaSelfAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states, head_mask=head_mask, output_attentions=output_attentions
            )

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            self.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=self.config.is_causal,
            scale=None,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = self.proj(context_layer.view(new_context_layer_shape))

        return context_layer, None


class VJEPASdpaRopeSelfAttention(VJEPARopeSelfAttention):
    def __init__(self, config: VJEPAConfig) -> None:
        super().__init__(config)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "VJEPAModel is using VJEPASdpaSelfAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states, head_mask=head_mask, output_attentions=output_attentions
            )

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        pos_ids = self.get_position_ids(hidden_states.device)
        key_layer = self.apply_rotary_embeddings(key_layer, pos_ids)
        query_layer = self.apply_rotary_embeddings(query_layer, pos_ids)

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            self.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=self.config.is_causal,
            scale=None,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = self.proj(context_layer.view(new_context_layer_shape))

        return context_layer, None


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->Dinov2->VJEPA
class VJEPASelfOutput(nn.Module):
    """
    The residual connection is defined in VJEPALayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: VJEPAConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->Dinov2->VJEPA
class VJEPAAttention(nn.Module):
    def __init__(self, config: VJEPAConfig) -> None:
        super().__init__()
        self.attention = VJEPASelfAttention(config)
        self.output = VJEPASelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class VJEPARopeAttention(nn.Module):
    def __init__(self, config: VJEPAConfig) -> None:
        super().__init__()
        self.attention = VJEPARopeSelfAttention(config)
        self.output = VJEPASelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTSdpaAttention with ViT->Dinov2
class VJEPASdpaAttention(VJEPAAttention):
    def __init__(self, config: VJEPAConfig) -> None:
        super().__init__(config)
        self.attention = VJEPASdpaSelfAttention(config)


class VJEPASdpaRopeAttention(VJEPAAttention):
    def __init__(self, config: VJEPAConfig) -> None:
        super().__init__(config)
        self.attention = VJEPASdpaRopeSelfAttention(config)


# Copied from transformers.models.beit.modeling_dinov2.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath
class VJEPADropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class VJEPAMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class VJEPASwiGLUFFN(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        if config.wide_SiLU:
            hidden_features = int(config.hidden_size * config.mlp_ratio)
            hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        else:
            hidden_features = int(config.hidden_size * config.mlp_ratio)

        self.weights_in = nn.Linear(in_features, 2 * hidden_features, bias=True)
        self.weights_out = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.weights_in(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = nn.functional.silu(x1) * x2
        return self.weights_out(hidden)


VJEPA_ATTENTION_CLASSES = {
    "eager": VJEPAAttention,
    "sdpa": VJEPASdpaAttention,
}

VJEPA_ROPE_ATTENTION_CLASSES = {
    "eager": VJEPARopeAttention,
    "sdpa": VJEPASdpaRopeAttention,
}


class VJEPALayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: VJEPAConfig, drop_path_rate: float = 0.0) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.use_sdpa:
            self.attention = VJEPASdpaRopeSelfAttention(config) if config.use_rope else VJEPASdpaSelfAttention(config)
        else:
            self.attention = (
                VJEPA_ROPE_ATTENTION_CLASSES[config._attn_implementation](config)
                if config.use_rope
                else VJEPA_ATTENTION_CLASSES[config._attn_implementation](config)
            )
        # self.layer_scale1 = Dinov2LayerScale(config)
        self.drop_path = VJEPADropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.use_SiLU:
            self.mlp = VJEPASwiGLUFFN(config)
        else:
            self.mlp = VJEPAMLP(config)
        # self.layer_scale2 = Dinov2LayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # in Dinov2, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        # attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in Dinov2, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        # layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->Dinov2->VJEPA
class VJEPAEncoder(nn.Module):
    def __init__(self, config: VJEPAConfig) -> None:
        super().__init__()
        self.config = config
        dpr = [
            x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)
        ]  # stochastic depth decay rule
        self.layer = nn.ModuleList([VJEPALayer(config, dpr[i]) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class VJEPAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VJEPAConfig
    base_model_prefix = "vjepa"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VJEPASwiGLUFFN"]
    _supports_sdpa = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, VJEPAEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            # Note: no CLS token in VJEPA
            # module.cls_token.data = nn.init.trunc_normal_(
            #     module.cls_token.data.to(torch.float32),
            #     mean=0.0,
            #     std=self.config.initializer_range,
            # ).to(module.cls_token.dtype)


# The main model
class VJEPAModel(VJEPAPreTrainedModel):
    def __init__(self, config: VJEPAConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = VJEPAEmbeddings(config)
        self.encoder = VJEPAEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> Union[VJEPAPatchEmbeddings, VJEPAPatchEmbeddings3D]:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = None  # there is no CLS tokens in VJEPA

        if not return_dict:
            head_outputs = (sequence_output, pooled_output)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
