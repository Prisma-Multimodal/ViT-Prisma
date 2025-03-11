# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import io
import json
import logging
import multiprocessing
import random
from dataclasses import dataclass
from itertools import islice
from multiprocessing import Value

import braceexpand
import webdataset as wds
from decord import VideoReader, cpu
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import tarfile_to_samples

VIDEO_EXTENSION_MP4 = "mp4"


multiprocessing.set_start_method("spawn", force=True)


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards_list):
    return len(shards_list)


def filter_video(sample: dict):
    return VIDEO_EXTENSION_MP4 in sample


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


class detshuffle(wds.PipelineStage):
    """Siinstitutionr to wds.detshuffle(), except here `epoch` is a SharedEpoch object instead instead of an int"""

    def __init__(
        self,
        bufsize=1000,
        initial=100,
        epoch=-1,
        seed=0,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        epoch = self.epoch.get_value()
        rng = random.Random()
        seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class split_by_node(wds.PipelineStage):
    """Node splitter that uses provided rank/world_size instead of from torch.distributed"""

    def __init__(
        self,
        rank=0,
        world_size=1,
    ):
        self.rank = rank
        self.world_size = world_size

    def run(self, src):
        if self.world_size > 1:
            yield from islice(src, self.rank, None, self.world_size)
        else:
            yield from src


class SimpleVideoDecoder:
    def __init__(
        self,
        frames_per_clip,
        duration=None,
        transform=None,
        shared_transform=None,
        pad_frames: bool = False,
    ):
        self.frames_per_clip = frames_per_clip
        self.duration = duration
        self.transform = transform
        self.shared_transform = shared_transform
        self.pad_frames = pad_frames

    def __call__(self, path):
        fpc = self.frames_per_clip
        all_indices = list(range(fpc))

        vr = VideoReader(path, num_threads=-1, ctx=cpu(0))
        if self.pad_frames:
            seek_indices = list(range(len(vr)))
        if not self.pad_frames:
            seek_indices = all_indices
            assert len(vr) >= fpc, f"Expect {fpc} frames got {len(vr)} frames"
        vr.seek(0)  # Go to start of video before sampling frames
        buffer = vr.get_batch(seek_indices).asnumpy()

        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        if self.transform is not None:
            buffer = self.transform(buffer)

        if self.pad_frames:
            if buffer.shape[1] != fpc:
                raise RuntimeError(
                    f"Expected {fpc} in idx 1 but found {buffer.shape}. It is possible that you need "
                    "to add a frame padding transform."
                )

        return [buffer], all_indices


class VideoSampleDecoder(object):
    """
    A generic wds sample decoder for video data.
    """

    def __init__(self, video_decoder=None, caption_preprocess=None):
        self.video_decoder = video_decoder
        self.caption_preprocess = caption_preprocess

    def __call__(self, sample):
        # TODO: As needed, re-add logic for processing of text (or text tokens).
        metadata = json.loads(sample["json"])
        if "video" in metadata:
            metadata = metadata["video"]
        annotation = metadata.get("annotation", "")
        key = metadata.get("__key__", metadata.get("__url__"))

        try:
            with io.BytesIO(sample[VIDEO_EXTENSION_MP4]) as stream:
                videos, frame_indices = self.video_decoder(stream)
        except Exception as e:
            log_and_continue(f"Failed to load sample: {key} ({e})")
            return None

        return {
            "video": videos,
            "indices": frame_indices,
            "text": annotation,
        }


def get_video_wds_dataset(
    batch_size,
    collator,
    input_shards,
    video_decoder,
    ipe=None,
    epoch=0,
    rank=0,
    world_size=1,
    num_workers=1,
    add_data_key=False,
):
    assert input_shards is not None
    num_shards = get_dataset_size(input_shards)
    logging.info(f"Total number of shards across all data is {num_shards=}")

    video_sample_decoder = VideoSampleDecoder(video_decoder=video_decoder)

    tuple_keys = ["video", "indices", "text"]
    if add_data_key:
        tuple_keys.append("__key__")

    epoch = SharedEpoch(epoch=epoch)
    pipeline = [
        wds.SimpleShardList(input_shards),
        detshuffle(bufsize=num_shards, initial=num_shards, epoch=epoch),
        split_by_node(rank=rank, world_size=world_size),
        wds.split_by_worker,
        # at this point, we have an iterator over the shards assigned to each worker at each node
        tarfile_to_samples(),
        detshuffle(bufsize=10000, initial=20, epoch=epoch),
        wds.select(filter_video),
        wds.map(video_sample_decoder, handler=log_and_continue),
        wds.to_tuple(*tuple_keys),
        wds.batched(batch_size, partial=False, collation_fn=collator),
    ]
    dataset = wds.DataPipeline(*pipeline)

    if ipe is not None and ipe > 0:
        assert ipe % max(num_workers, 1) == 0, "num_workers should evenly divide # iterations per epoch"
        logging.info(f"Setting epoch iterations {ipe}")
        dataset = dataset.with_epoch(ipe // max(num_workers, 1))

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        timeout=0,
        pin_memory=True,
    )

    dataloader.num_batches = ipe

    return dataset, DataInfo(dataloader=dataloader, shared_epoch=epoch)


def make_video_webdataset(
    data_paths,
    batch_size,
    transform,
    shared_transform,
    ipe=300,
    collator=None,
    num_frames=16,
    duration=None,
    rank=0,
    num_workers=8,
    world_size=1,
    add_data_key=False,
    input_shards=None,
    pad_frames: bool = False,
):
    if input_shards is None:
        # Each element of data_paths is a glob that needs to be expanded
        input_shards = [path for glob in data_paths for path in braceexpand.braceexpand(glob)]
    else:
        assert type(input_shards) is list
        assert len(input_shards) >= 1, "At least one shard should be provided"

    video_decoder = SimpleVideoDecoder(
        frames_per_clip=num_frames,
        transform=transform,
        shared_transform=shared_transform,
        duration=duration,  # duration in seconds
        pad_frames=pad_frames,
    )

    dataset, datainfo = get_video_wds_dataset(
        batch_size=batch_size,
        collator=collator,
        input_shards=input_shards,
        ipe=ipe,
        epoch=0,
        rank=rank,
        world_size=world_size,
        num_workers=num_workers,
        video_decoder=video_decoder,
        add_data_key=add_data_key,
    )

    return dataset, datainfo.dataloader, datainfo
