import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import lance
import mne

from configs import Config
from dataset import preprocess_all_samples


def load_edf_as_raw(edf_path: str):
    """
    读取一个 EDF 文件，返回 MNE Raw 对象。
    """
    edf_path = str(edf_path)

    if not Path(edf_path).exists():
        raise FileNotFoundError(f"Cannot find EDF file: {edf_path}")

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose="ERROR",
    )

    return raw


def split_raw_to_fixed_length_samples(
    raw,
    source_file: str,
    dataset_name: str,
    clip_seconds: float,
    clip_stride_seconds: float,
    convert_v_to_uv: bool = True,
):
    """
    把连续 EDF Raw 切成固定长度 samples。

    Returns:
        samples: list[dict]
        每个 sample:
            {
                "signal": [C, T],
                "channel_names": list[str],
                "sfreq": float,
                "dataset_name": str,
                "source_file": str,
                "clip_idx": int,
                "clip_start_sec": float,
            }
    """
    signal = raw.get_data().astype(np.float32)  # [C, T]

    if convert_v_to_uv:
        signal = signal * 1e6

    channel_names = list(raw.ch_names)
    sfreq = float(raw.info["sfreq"])

    clip_len = int(round(clip_seconds * sfreq))
    stride_len = int(round(clip_stride_seconds * sfreq))

    if clip_len <= 0:
        raise ValueError("clip_len must be positive.")

    if stride_len <= 0:
        raise ValueError("stride_len must be positive.")

    total_points = signal.shape[1]

    samples = []

    start = 0
    clip_idx = 0

    while start + clip_len <= total_points:
        end = start + clip_len

        clip_signal = signal[:, start:end].astype(np.float32)

        samples.append({
            "signal": clip_signal,
            "channel_names": channel_names,
            "sfreq": sfreq,
            "dataset_name": dataset_name,
            "source_file": source_file,
            "clip_idx": clip_idx,
            "clip_start_sec": start / sfreq,
        })

        start += stride_len
        clip_idx += 1

    return samples


def load_multiple_edfs_as_samples(
    edf_paths,
    dataset_name: str,
    clip_seconds: float,
    clip_stride_seconds: float,
    convert_v_to_uv: bool = True,
):
    """
    读取多个 EDF 文件，并合并成一个 samples list。
    """
    all_samples = []

    for edf_path in edf_paths:
        edf_path = str(edf_path)
        print(f"Loading EDF: {edf_path}")

        raw = load_edf_as_raw(edf_path)

        samples = split_raw_to_fixed_length_samples(
            raw=raw,
            source_file=edf_path,
            dataset_name=dataset_name,
            clip_seconds=clip_seconds,
            clip_stride_seconds=clip_stride_seconds,
            convert_v_to_uv=convert_v_to_uv,
        )

        print(f"  created {len(samples)} clips")

        all_samples.extend(samples)

    return all_samples


def collect_edf_paths(args, cfg):
    """
    收集 EDF 文件路径。
    优先级:
        1. 命令行 --edf_paths
        2. 命令行 --edf_dir
        3. cfg.data.edf_paths
    """
    edf_paths = []

    if args.edf_paths:
        edf_paths.extend(args.edf_paths)

    if args.edf_dir:
        edf_dir = Path(args.edf_dir)
        edf_paths.extend(sorted(str(x) for x in edf_dir.rglob("*.edf")))

    if len(edf_paths) == 0 and hasattr(cfg.data, "edf_paths"):
        edf_paths.extend(cfg.data.edf_paths)

    edf_paths = sorted(set(str(x) for x in edf_paths))

    return edf_paths


def flatten_sample(sample):
    return {
        "token_inputs": sample["token_inputs"].reshape(-1).astype(np.float32).tolist(),
        "targets": sample["targets"].reshape(-1).astype(np.float32).tolist(),
        "token_channel_indices": sample["token_channel_indices"].astype(np.int64).tolist(),
        "token_time_indices": sample["token_time_indices"].astype(np.int64).tolist(),
        "token_valid_mask": sample["token_valid_mask"].astype(np.float32).tolist(),
        "channel_valid_mask": sample["channel_valid_mask"].astype(np.float32).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="data/pretrain_processed_edf.lance",
        help="输出 Lance 路径",
    )

    parser.add_argument(
        "--edf_paths",
        nargs="*",
        default=None,
        help="一个或多个 EDF 文件路径",
    )

    parser.add_argument(
        "--edf_dir",
        default=None,
        help="包含 EDF 文件的文件夹，会递归搜索 *.edf",
    )

    parser.add_argument(
        "--dataset_name",
        default="edf_dataset",
        help="数据集名称，会写入 Lance metadata",
    )

    args = parser.parse_args()

    cfg = Config()

    edf_paths = collect_edf_paths(args, cfg)

    print("===== EDF Files =====")
    for p in edf_paths[:20]:
        print(p)

    if len(edf_paths) > 20:
        print(f"... and {len(edf_paths) - 20} more")

    if len(edf_paths) == 0:
        raise ValueError(
            "No EDF files found. Please pass --edf_paths, --edf_dir, "
            "or define cfg.data.edf_paths."
        )

    print("===== Loading Raw EDF Clips =====")

    raw_samples = load_multiple_edfs_as_samples(
        edf_paths=edf_paths,
        dataset_name=args.dataset_name,
        clip_seconds=cfg.data.clip_seconds,
        clip_stride_seconds=cfg.data.clip_stride_seconds,
        convert_v_to_uv=cfg.data.convert_v_to_uv,
    )

    print(f"Total raw clips: {len(raw_samples)}")

    if len(raw_samples) == 0:
        raise ValueError("No raw clips found.")

    print("===== Preprocessing All Clips =====")

    processed_samples = preprocess_all_samples(raw_samples, cfg)

    if len(processed_samples) == 0:
        raise ValueError("No processed samples generated.")

    if len(processed_samples) != len(raw_samples):
        raise ValueError(
            f"processed_samples length mismatch: "
            f"{len(processed_samples)} vs raw_samples {len(raw_samples)}. "
            f"If preprocess_all_samples skips samples, metadata alignment needs to be handled explicitly."
        )

    first = processed_samples[0]

    token_inputs_shape = first["token_inputs"].shape
    targets_shape = first["targets"].shape
    token_channel_indices_shape = first["token_channel_indices"].shape
    token_time_indices_shape = first["token_time_indices"].shape
    token_valid_mask_shape = first["token_valid_mask"].shape
    channel_valid_mask_shape = first["channel_valid_mask"].shape

    print("===== First Processed Sample Shape =====")
    print("token_inputs:", token_inputs_shape)
    print("targets:", targets_shape)
    print("token_channel_indices:", token_channel_indices_shape)
    print("token_time_indices:", token_time_indices_shape)
    print("token_valid_mask:", token_valid_mask_shape)
    print("channel_valid_mask:", channel_valid_mask_shape)

    num_tokens, patch_len = token_inputs_shape
    _, n_bands, frames_per_patch = targets_shape
    n_channels = channel_valid_mask_shape[0]

    rows = []

    for i, sample in enumerate(processed_samples):
        raw_meta = raw_samples[i]

        if sample["token_inputs"].shape != token_inputs_shape:
            raise ValueError(f"sample {i} token_inputs shape mismatch")

        if sample["targets"].shape != targets_shape:
            raise ValueError(f"sample {i} targets shape mismatch")

        row = flatten_sample(sample)

        row.update({
            "sample_idx": i,
            "global_idx": i,

            "dataset_name": raw_meta.get("dataset_name", args.dataset_name),
            "source_file": raw_meta.get("source_file", ""),
            "clip_idx": int(raw_meta.get("clip_idx", -1)),
            "clip_start_sec": float(raw_meta.get("clip_start_sec", -1.0)),

            "num_tokens": num_tokens,
            "patch_len": patch_len,
            "n_bands": n_bands,
            "frames_per_patch": frames_per_patch,
            "n_channels": n_channels,
        })

        rows.append(row)

    schema = pa.schema([
        pa.field("sample_idx", pa.int64()),
        pa.field("global_idx", pa.int64()),

        pa.field("dataset_name", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("clip_idx", pa.int64()),
        pa.field("clip_start_sec", pa.float32()),

        pa.field("num_tokens", pa.int64()),
        pa.field("patch_len", pa.int64()),
        pa.field("n_bands", pa.int64()),
        pa.field("frames_per_patch", pa.int64()),
        pa.field("n_channels", pa.int64()),

        pa.field("token_inputs", pa.list_(pa.float32(), num_tokens * patch_len)),
        pa.field("targets", pa.list_(pa.float32(), num_tokens * n_bands * frames_per_patch)),
        pa.field("token_channel_indices", pa.list_(pa.int64(), num_tokens)),
        pa.field("token_time_indices", pa.list_(pa.int64(), num_tokens)),
        pa.field("token_valid_mask", pa.list_(pa.float32(), num_tokens)),
        pa.field("channel_valid_mask", pa.list_(pa.float32(), n_channels)),
    ])

    table = pa.Table.from_pylist(rows, schema=schema)

    print(f"===== Writing Lance Dataset: {args.output} =====")

    lance.write_dataset(
        table,
        args.output,
        mode="overwrite",
    )

    print("Done.")
    print(f"Wrote {len(rows)} samples to {args.output}")


if __name__ == "__main__":
    main()