# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from logging import getLogger
from multiprocessing import Value

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        enc_sparsity_factor=0.5,
        npred=2,
        min_keep=10,
        allow_overlap=False,
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.esf = enc_sparsity_factor
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self._itr_counter = Value("i", -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()

        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)

        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size):
        h, w = b_size

        # -- Sample block top-left corner
        # Fix fencepost error: torch.randint(low, high) has an exclusive upper bound.
        # Using self.height - h + 1 ensures the final row and column indices can be sampled.
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        mask = torch.zeros((self.height, self.width), dtype=torch.int32)
        mask[top : top + h, left : left + w] = 1
        mask = torch.nonzero(mask.flatten())
        mask = mask.squeeze()
        mask = mask[torch.randperm(len(mask))]
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top : top + h, left : left + w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample pred block (size) using seed
        # 2. sample several pred block locations for each image (w/o seed)
        # 3. return complement mask and pred mask
        """
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(generator=g, scale=self.pred_mask_scale, aspect_ratio_scale=self.aspect_ratio)

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):

            masks_p, complement_mask = [], None
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                if complement_mask is None:
                    complement_mask = mask_C
                complement_mask *= mask_C
                masks_p.append(mask)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            complement_mask = torch.nonzero(complement_mask.flatten())
            complement_mask = complement_mask.squeeze()
            complement_mask = complement_mask[torch.randperm(len(complement_mask))]
            min_keep_enc = min(min_keep_enc, len(complement_mask))
            collated_masks_enc.append([complement_mask])

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        min_keep_enc = int(max(self.min_keep, min_keep_enc) * self.esf)
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred
