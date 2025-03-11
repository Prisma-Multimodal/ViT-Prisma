import math
import os
import random
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from airstore.blobstore import BlobStore, S3ServerSharedMode
from botocore.exceptions import ClientError
from decord import VideoReader, cpu

from vit_prisma.vjepa_hf.src.datasets.utils.dataloader import NondeterministicDataLoader
from vit_prisma.vjepa_hf.src.datasets.utils.weighted_sampler import DistributedWeightedSampler
from vit_prisma.vjepa_hf.src.utils.logging import get_logger
from vit_prisma.vjepa_hf.src.utils.profiler import create_profiler

logger = get_logger("BlobStore Data Loader")


VIDEO_PATH_DB = {
    "yttemporal1b_512": "/checkpoint/jepa/data/blobstore_index/yttemporal1b_512.db",
    "yttemporal1b_360px_120s": "/checkpoint/jepa/data/blobstore_index/yttemporal1b_res360_chunklength120_partial.db",
    "yttemporal1b_resized/res360_chunklength120": "/checkpoint/jepa/data/blobstore_index/yttemporal1b_resized360p_chunked120.db",  # NOQA
    "yttemporal1b_resized/res480_chunklength120": "/checkpoint/jepa/data/blobstore_index/yttemporal1b_resized480p_chunked120.db",  # NOQA
    "yttemporal1b_resized/res720_chunklength120": "/checkpoint/jepa/data/blobstore_index/yttemporal1b_resized720p_chunked120.db",  # NOQA
    # The first attempt on data curation from YT1B from Kinetics.
    # The requsted k factor is 10, but we ended up with ~0.115 (84,097 examples)
    "yt1b_kinetics_curated_k10_v1": "/checkpoint/jepa/data/blobstore_index/yt1b_kinetics_curated_k10_v1.db",
    "howto100m": "/checkpoint/jepa/data/blobstore_index/howto100m.db",
    "howto100m_resized/res360_chunklength120": "/checkpoint/jepa/data/blobstore_index/howto100m_resized360p_chunked120.db",  # NOQA
    "ssv2": "/checkpoint/jepa/data/blobstore_index/ssv2.db",
    "jepa_kinetics_710_train": "/checkpoint/jepa/data/blobstore_index/jepa_kinetics_710_train.db",
    "ego4d_track2_v1_full_512": "/checkpoint/jepa/data/blobstore_index/ego4d_track2_v1_full_512.db",
    "ego4d_chunked_512": "/checkpoint/jepa/data/blobstore_index/ego4d_chunked_512.db",
    "ego4d_full_length": "/checkpoint/jepa/data/blobstore_index/ego4d_full_length.db",
    "ego4d_resized/res360_chunklength120": "/checkpoint/jepa/data/blobstore_index/ego4d_resized360p_chunked120.db",
    "ego4d_resized/res480_chunklength120": "/checkpoint/jepa/data/blobstore_index/ego4d_resized480p_chunked120.db",
    "ego4d_resized/res720_chunklength120": "/checkpoint/jepa/data/blobstore_index/ego4d_resized720p_chunked120.db",
}

CHUNKED_VIDEO_PATH_DB = {
    "yttemporal1b_resized/res360_chunklength120": "/checkpoint/jepa/data/blobstore_index/yttemporal1b_resized360p_chunked120_chunklist.db",  # NOQA
    "yttemporal1b_resized/res480_chunklength120": "/checkpoint/jepa/data/blobstore_index/yttemporal1b_resized480p_chunked120_chunklist.db",  # NOQA
    "yttemporal1b_resized/res720_chunklength120": "/checkpoint/jepa/data/blobstore_index/yttemporal1b_resized720p_chunked120_chunklist.db",  # NOQA
    "howto100m_resized/res360_chunklength120": "/checkpoint/jepa/data/blobstore_index/howto100m_resized360p_chunked120_chunklist.db",  # NOQA
    "ego4d_resized/res360_chunklength120": "/checkpoint/jepa/data/blobstore_index/ego4d_resized360p_chunked120_chunklist.db",  # NOQA
    "ego4d_resized/res480_chunklength120": "/checkpoint/jepa/data/blobstore_index/ego4d_resized480p_chunked120_chunklist.db",  # NOQA
    "ego4d_resized/res720_chunklength120": "/checkpoint/jepa/data/blobstore_index/ego4d_resized720p_chunked120_chunklist.db",  # NOQA
}


# We are caching this because it is not an object that could be pickled.
# Therefore the dataloader can not pickle it for multiple workers.
# Getting the db connection every time and caching that at this level avoids the pickling issue.
@lru_cache
def get_db(fpath):
    logger.info(f"Connecting to the captions DB at {fpath}")
    db = sqlite3.connect(f"file:{fpath}?mode=ro", uri=True)
    logger.info("Captions DB connection successful.")
    return db


@lru_cache
def temp_ssd_dir():
    job_id = os.environ.get("SLURM_JOB_ID", None)
    if job_id:
        slurm_ssd_temp_path = os.path.join("/scratch/slurm_tmpdir", job_id)
        if os.path.exists(slurm_ssd_temp_path):
            logger.info(f"Using existing slurm temp dir: {slurm_ssd_temp_path}")
            return slurm_ssd_temp_path
    logger.info("Using the default OS path for temp dir")


@dataclass
class DBEntry:
    db_path: str
    count: int


def get_example_paths_from_blobstore(dataset_path):
    with BlobStore() as blobstore:
        blobstore.start()
        examples = []
        continuation_token = None
        logger.info(f"Getting examples for {dataset_path}. This may take a while ...")
        while True:
            result = blobstore.list_objects(
                bucket="jepa",
                path=dataset_path,
                continuation_token=continuation_token,
                max_keys=5000,
            )
            assert not result.blobs, (
                "The conventions is that each examples is inside a directory:"
                " there MUST be no blobs at the top level."
            )
            examples.extend(result.directories)
            if result.continuation_token is None:
                break
            continuation_token = result.continuation_token

    logger.info(f"Found {len(examples)} examples in {dataset_path}")
    return examples


def get_num_examples_from_sqlite_db(dataset_path):
    logger.info(f"Getting number of examples from {dataset_path}.")
    db = get_db(dataset_path)
    query = "SELECT COUNT(*) FROM video_ids"
    response = db.execute(query)
    assert response is not None, "There must be one row of video ids."
    count = next(response)[0]
    assert isinstance(count, int), "The number of videos must be an integer."
    logger.info(f"Found {count} examples in {dataset_path}")
    return count


def get_item_from_rowid(db, row_id):
    query = f"SELECT video_path FROM video_ids WHERE ROWID={row_id}"
    response = db.execute(query)
    assert response is not None, "There must be one row of video ids."
    video_id = next(response)[0]
    assert isinstance(video_id, str), "The video id must be a string."
    return video_id


def get_chunked_vid_key(db, row_id):
    query = f"SELECT blobstore_key FROM video_ids WHERE ROWID={row_id}"
    response = db.execute(query)
    video_id = response.fetchone()[0]

    if video_id is None:
        raise RuntimeError(f"Failed to get video path for row id {row_id}")

    if not isinstance(video_id, str):
        raise TypeError(f"Video path is not a string: {video_id}")

    return video_id


def copy_path_db_to_local(ds, local_dir, chunked_db: bool = False):
    if chunked_db:
        db_path = CHUNKED_VIDEO_PATH_DB[ds]
    else:
        db_path = VIDEO_PATH_DB[ds]
    # Copy the DB to local storage to avoid NFS overhead (after checking no name collisions)
    basename = os.path.basename(db_path)
    local_db_path = os.path.join(local_dir, basename)
    assert not os.path.exists(local_db_path), f"DB file {basename} already exists locally. Duplicate name?"
    shutil.copyfile(db_path, local_db_path)
    return local_db_path


def make_blobstore_dataset(
    data_paths,
    batch_size,
    split="train",
    dataset_fpcs=None,
    frames_per_clip=None,
    frame_step=None,
    duration=None,
    fps=4,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    transform=None,
    shared_transform=None,
    world_size=1,
    rank=0,
    log_dir=None,
    datasets_weights=None,
    num_workers=8,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    drop_last=True,
    deterministic=True,
    return_dict=False,
    dataloader_profiler_conf=None,
    chunked_db: bool = False,
):
    if chunked_db:
        ds_class = ChunkedBlobStoreDataset
    else:
        ds_class = BlobStoreDataset

    dataset = ds_class(
        data_paths,
        split=split,
        transform=transform,
        shared_transform=shared_transform,
        fps=fps,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        world_size=world_size,
        rank=rank,
        log_dir=log_dir,
        datasets_weights=datasets_weights,
        duration=duration,
        return_dict=return_dict,
        profiler_conf=dataloader_profiler_conf,
    )

    logger.info(f"Dataset in {split} split for rank {rank} / {world_size} created.")
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    if deterministic:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    else:
        data_loader = NondeterministicDataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    logger.info("BlobStore dataset data loader created")

    return dataset, data_loader, dist_sampler


class BlobStoreDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_paths,
        split="train",
        transform=None,
        shared_transform=None,
        frames_per_clip=16,
        fps=None,
        frame_step=None,
        num_clips=1,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        world_size=1,
        rank=0,
        log_dir=None,
        datasets_weights=None,
        dataset_fpcs=None,
        duration=None,  # duration in seconds
        return_dict=False,
        profiler_conf=None,
    ):
        if (int(fps is not None) + int(duration is not None) + int(frame_step is not None)) == 0:
            raise ValueError("Either fps or duration or frame_step must be specified")

        elif (int(fps is not None) + int(duration is not None) + int(frame_step is not None)) > 1:
            raise ValueError("Only one of fps or duration or frame_step can be specified")

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if dataset_fpcs is None:
            dataset_fpcs = [frames_per_clip for _ in data_paths]

        assert len(dataset_fpcs) == len(data_paths), "Frames per clip not properly specificed for Blobstore data paths"

        self.fps = fps
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.duration = duration
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap

        self.data_paths = data_paths
        self.split = split
        self.transform = transform
        self.shared_transform = shared_transform
        self.world_size = world_size
        self.rank = rank
        self.log_dir = log_dir
        self.return_dict = return_dict
        self.profiler = None
        self.profiler_conf = profiler_conf
        self.local_db_dir = tempfile.mkdtemp()

        self.samples_chunks = []
        self.num_samples_per_dataset = []
        for ds in self.data_paths:
            """
            We mix two types of datasets here (in both cases we have all the samples in BlobStore):
            1) There is not an index of samples as a SQLite DB. In this case we get the samples from BlobStore
                directly. This can be very slow due to the lack of parallelism in the BlobStore API. It may also
                exhaust the memory because of the large number of samples, which we now keep as a list in memory.
            2) [preferred] There is an index of samples as a SQLite DB. In this case we only get the sample count
                from the SQLite DB. Later we take the index into DB (row_id) for retrieving the BlobStore path of
                that sample from our DB.
            """
            if ds not in VIDEO_PATH_DB:
                logger.warning(
                    f"Dataset {ds} doesn't have an index db file. "
                    "This makes initialization extremely slow. "
                    "Consider creating the SQLite DB for the index of this dataset (contact Mojtaba)."
                )
                examples = get_example_paths_from_blobstore(ds)
                self.samples_chunks.append(examples)
                self.num_samples_per_dataset.append(len(examples))
            else:
                db_path = copy_path_db_to_local(
                    ds, self.local_db_dir, chunked_db=isinstance(self, ChunkedBlobStoreDataset)
                )
                num_examples = get_num_examples_from_sqlite_db(db_path)
                db_entry = DBEntry(db_path=db_path, count=num_examples)
                self.samples_chunks.append(db_entry)
                self.num_samples_per_dataset.append(num_examples)

        # [Optional] Weights for each sample to be used by downstream
        # weighted video sampler
        self.sample_weights = None
        if datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.dataset_fpcs = []
        for fpc, ns in zip(dataset_fpcs, self.num_samples_per_dataset):
            self.dataset_fpcs += [fpc] * ns

        self._blobstore_client = None

    def get_profiler(self, profiler_conf):
        if profiler_conf is None:
            return None

        if self.profiler is not None:
            # Already created a profiler
            return self.profiler

        profiler_log_dir = os.path.join(self.log_dir, "profiler")
        profiler = create_profiler(profiler_conf, profiler_log_dir, self.rank)
        profiler.start()
        return profiler

    def __del__(self) -> None:
        try:
            if self._blobstore_client is not None:
                self._blobstore_client.cleanup()
            shutil.rmtree(self.local_db_dir, ignore_errors=True)
        except AttributeError:
            pass

    def __len__(self):
        total = 0
        for chunk in self.samples_chunks:
            if isinstance(chunk, DBEntry):
                total += chunk.count
            else:
                assert isinstance(chunk, list)
                total += len(chunk)
        return total

    def load_video(self, video_path, fpc):
        try:
            vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
            vr.seek(0)  # Go to start of video before sampling frames
        except Exception:
            return None, None

        fstp = self.frame_step
        if self.duration is not None or self.fps is not None:
            try:
                video_fps = math.ceil(vr.get_avg_fps())
            except Exception as e:
                logger.warning(e)

            if self.duration is not None:
                assert self.fps is None
                fstp = int(self.duration * video_fps / fpc)
            else:
                assert self.duration is None
                fstp = video_fps // self.fps

        assert fstp is not None and fstp > 0
        clip_len = int(fpc * fstp)

        # Partition video into equal sized segments and sample each clip
        # from a different segment
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(fpc - partition_len // fstp) * partition_len,
                        )
                    )
                    indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(fpc - sample_len // fstp) * sample_len,
                        )
                    )
                    indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                    # --
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        try:
            buffer = vr.get_batch(all_indices).asnumpy()
        except Exception:
            return None, None

        return buffer, clip_indices

    def get_label(self, example):
        # TODO add label pulling later, either in AIRStore data or from matching with the annotation file.
        return 0

    def get_blobstore_client(self):
        if self._blobstore_client is None:
            self._blobstore_client = BlobStore(mode=S3ServerSharedMode.EXCLUSIVE)
            self._blobstore_client.start()
        return self._blobstore_client

    def get_video_blob(self, path):
        _ = self.get_blobstore_client()
        ret = self._blobstore_client.list_objects(
            bucket="jepa",
            path=path,
            continuation_token=None,
            max_keys=100,
        )
        if len(ret.blobs) == 0:
            raise RuntimeError(f"No blobs found at path {path}!")
        if len(ret.blobs) > 1:
            # Hacky workaround: leave out the last blob, which may be too short to yield a full clip
            # TODO: incorporate chunk boundary times into .db so we can select chunk(s) based on selected time
            blob = random.choice(ret.blobs[:-1])
        else:
            blob = ret.blobs[0]
        return self._blobstore_client.get(key=f"{path}/{blob.name}", bucket="jepa")

    def _get_blobstore_path(self, idx):
        """
        Gets the path of the video in BlobStore.

        Having multiple datasets, we keep the paths for each dataset in a chunk (self.samples_chunks).
        Generally, for smaller datasets the chunk is a list of paths, but for larger datasets we keep an
        DBEntry object that contains the DB connection and the number of samples in the dataset.
        A typical self.samples_chunks may look like this:

        [
            ['ds1/sydf6', ..., 'ds1/iouert65'] /* chunk 1 list */,
            <DBEntry object> /* chunk 2 DBEntry */,
            <DBEntry object> /* chunk 3 DBEntry */,
        ]

        When getting the path of the video, we first find the chunk that the video belongs to.
        Then find its index in the chunk and return the path by querying the DBEntry object or returning from the list.
        """
        # 1) find the index of the chunk that the video belongs to, and the index of the video in that chunk.
        chunk_start_idx = 0
        for chunk in self.samples_chunks:
            if isinstance(chunk, DBEntry):
                chunk_length = chunk.count
            else:
                # Probably an overkill to assert here multiple times,
                # specially since it happens in __len__ as well; but just in case.
                assert isinstance(chunk, list)
                chunk_length = len(chunk)

            chunk_end_idx = chunk_start_idx + chunk_length
            if idx < chunk_end_idx:
                break

            chunk_start_idx = chunk_end_idx

        # 2) return the path of the video in the chunk
        item_index_in_chunk = idx - chunk_start_idx
        if isinstance(chunk, DBEntry):
            db = get_db(chunk.db_path)
            db_row_id = item_index_in_chunk + 1  # row_id starts from 1
            return get_item_from_rowid(db, db_row_id)
        else:
            return chunk[item_index_in_chunk]

    def __getitem__(self, idx):
        sample = self._get_blobstore_path(idx)
        fpc = self.dataset_fpcs[idx]
        self.profiler = self.get_profiler(self.profiler_conf)

        buffer = None
        # Keep trying to load videos until you find a valid sample
        while buffer is None:
            temp_vid_file = tempfile.NamedTemporaryFile(dir=temp_ssd_dir())
            try:
                video_blob = self.get_video_blob(sample)
                temp_vid_file.write(video_blob)
                temp_vid_file.flush()
                buffer, clip_indices = self.load_video(temp_vid_file.name, fpc)
            except ClientError as e:
                # Fail out instead of suppressing access failure (e.g. missing crypto key)
                if "AccessDenied" in str(e):
                    raise e
            except Exception as e:
                buffer = None
                logger.warning(f"Failed to load video {sample}: {e}")
            finally:
                temp_vid_file.close()

            if buffer is None:
                idx = np.random.randint(self.__len__())
                sample = self._get_blobstore_path(idx)

        if self.shared_transform:
            buffer = self.shared_transform(buffer)
        # Splitting into n=self.num_clips clips
        buffer = [buffer[i * fpc : (i + 1) * fpc] for i in range(self.num_clips)]

        if self.transform:
            buffer = [self.transform(clip) for clip in buffer]

        label = self.get_label(sample)

        if self.profiler is not None:
            self.profiler.step()

        if self.return_dict:
            return {
                "clips": buffer,
                # "video_id": video_id,  # TODO: fix or remove
                "label": label,
                # the following fields are for debugging; can be removed later
                "job_rank": self.rank,
            }
        else:
            # This is the format used by the src.datasets.video_dataset.VideoDataset
            return buffer, label, clip_indices


class ChunkedBlobStoreDataset(BlobStoreDataset):
    def _get_blobstore_path(self, idx):
        """
        Siinstitutionr to parent class operation, but we retrieve the path directly from the chunk.
        """
        # 1) find the index of the chunk that the video belongs to, and the index of the video in that chunk.
        chunk_start_idx = 0
        for chunk in self.samples_chunks:
            if isinstance(chunk, DBEntry):
                chunk_length = chunk.count
            else:
                # Probably an overkill to assert here multiple times,
                # specially since it happens in __len__ as well; but just in case.
                assert isinstance(chunk, list)
                chunk_length = len(chunk)

            chunk_end_idx = chunk_start_idx + chunk_length
            if idx < chunk_end_idx:
                break

            chunk_start_idx = chunk_end_idx

        # 2) return the path of the video in the chunk
        item_index_in_chunk = idx - chunk_start_idx
        if isinstance(chunk, DBEntry):
            db = get_db(chunk.db_path)
            db_row_id = item_index_in_chunk + 1  # row_id starts from 1
            return get_chunked_vid_key(db, db_row_id)
        else:
            return chunk[item_index_in_chunk]

    def get_video_blob(self, path):
        cli = self.get_blobstore_client()
        return cli.get(key=path, bucket="jepa")
