import argparse
import json
from pathlib import Path

import lance
import lancedb
import numpy as np
import pyarrow as pa

from configs import Config
from dataset import preprocess_all_samples


def load_id2name(electrode_vocab_path: str) -> dict[int, str]:
    with open(electrode_vocab_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def electrode_ids_to_channel_names(eids: np.ndarray, id2name: dict[int, str]) -> list[str]:
    names = []
    for eid in eids.tolist():
        eid = int(eid)
        if eid == 0:
            names.append("PAD")
        else:
            names.append(id2name.get(eid, "UNK_EEG"))
    return names


def flatten_sample(sample):
    return {
        "token_inputs": sample["token_inputs"].reshape(-1).astype(np.float32).tolist(),
        "targets": sample["targets"].reshape(-1).astype(np.float32).tolist(),
        "token_channel_indices": sample["token_channel_indices"].astype(np.int64).tolist(),
        "token_time_indices": sample["token_time_indices"].astype(np.int64).tolist(),
        "token_valid_mask": sample["token_valid_mask"].astype(np.float32).tolist(),
        "channel_valid_mask": sample["channel_valid_mask"].astype(np.float32).tolist(),
    }


def make_schema_from_first_processed(first):
    token_inputs_shape = first["token_inputs"].shape
    targets_shape = first["targets"].shape
    token_channel_indices_shape = first["token_channel_indices"].shape
    token_time_indices_shape = first["token_time_indices"].shape
    token_valid_mask_shape = first["token_valid_mask"].shape
    channel_valid_mask_shape = first["channel_valid_mask"].shape

    num_tokens, patch_len = token_inputs_shape
    _, n_bands, frames_per_patch = targets_shape
    n_channels = channel_valid_mask_shape[0]

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

    shape_meta = {
        "num_tokens": num_tokens,
        "patch_len": patch_len,
        "n_bands": n_bands,
        "frames_per_patch": frames_per_patch,
        "n_channels": n_channels,
        "token_inputs_shape": token_inputs_shape,
        "targets_shape": targets_shape,
        "token_channel_indices_shape": token_channel_indices_shape,
        "token_time_indices_shape": token_time_indices_shape,
        "token_valid_mask_shape": token_valid_mask_shape,
        "channel_valid_mask_shape": channel_valid_mask_shape,
    }

    return schema, shape_meta


def foundation_batch_to_raw_samples(
    batch,
    *,
    id2name: dict[int, str],
    n_channels: int,
    seq_len: int,
    dataset_name: str,
    denorm_clip_divide_200: bool = True,
):
    B = batch.num_rows

    sig_col = batch.column("signals").combine_chunks()
    sig_flat = sig_col.values.to_numpy(zero_copy_only=False)
    signals = sig_flat.reshape(B, n_channels, seq_len).astype(np.float32, copy=True)

    if denorm_clip_divide_200:
        # foundation 表里的 signals 通常已经是 clip_divide_200 后的值；
        # 这里乘回 μV，保持和 convert_set_to_lance 原始流程一致。
        signals *= 200.0

    eid_col = batch.column("electrode_ids").combine_chunks()
    eid_flat = eid_col.values.to_numpy(zero_copy_only=False)
    electrode_ids = eid_flat.reshape(B, n_channels).astype(np.int64, copy=True)

    sample_ids = batch.column("sample_id").to_pylist() if "sample_id" in batch.schema.names else list(range(B))
    global_indices = batch.column("global_idx").to_pylist() if "global_idx" in batch.schema.names else sample_ids
    subject_ids = batch.column("subject_id").to_pylist() if "subject_id" in batch.schema.names else ["unknown"] * B
    edf_relpaths = batch.column("edf_relpath").to_pylist() if "edf_relpath" in batch.schema.names else [""] * B
    starts = batch.column("segment_start_sec").to_pylist() if "segment_start_sec" in batch.schema.names else [0.0] * B
    channel_counts = batch.column("channel_counts").to_pylist() if "channel_counts" in batch.schema.names else [n_channels] * B

    raw_samples = []
    raw_meta = []

    for i in range(B):
        n_ch = int(channel_counts[i])
        n_ch = min(n_ch, n_channels)

        sig = signals[i, :n_ch, :]
        eids = electrode_ids[i, :n_ch]
        ch_names = electrode_ids_to_channel_names(eids, id2name)

        source_file = edf_relpaths[i] or str(subject_ids[i])

        raw_samples.append({
            "signal": sig,
            "channel_names": ch_names,
            "sfreq": 200.0,

            "dataset_name": dataset_name,
            "source_file": source_file,
            "clip_idx": int(sample_ids[i]),
            "clip_start_sec": float(starts[i]),
        })

        raw_meta.append({
            "sample_idx": int(sample_ids[i]),
            "global_idx": int(global_indices[i]),
            "dataset_name": dataset_name,
            "source_file": source_file,
            "clip_idx": int(sample_ids[i]),
            "clip_start_sec": float(starts[i]),
        })

    return raw_samples, raw_meta


def processed_to_rows(processed_samples, raw_meta, shape_meta):
    rows = []

    for i, sample in enumerate(processed_samples):
        if sample["token_inputs"].shape != shape_meta["token_inputs_shape"]:
            raise ValueError(f"sample {i} token_inputs shape mismatch: {sample['token_inputs'].shape}")

        if sample["targets"].shape != shape_meta["targets_shape"]:
            raise ValueError(f"sample {i} targets shape mismatch: {sample['targets'].shape}")

        row = flatten_sample(sample)
        meta = raw_meta[i]

        row.update({
            "sample_idx": int(meta["sample_idx"]),
            "global_idx": int(meta["global_idx"]),

            "dataset_name": str(meta["dataset_name"]),
            "source_file": str(meta["source_file"]),
            "clip_idx": int(meta["clip_idx"]),
            "clip_start_sec": float(meta["clip_start_sec"]),

            "num_tokens": int(shape_meta["num_tokens"]),
            "patch_len": int(shape_meta["patch_len"]),
            "n_bands": int(shape_meta["n_bands"]),
            "frames_per_patch": int(shape_meta["frames_per_patch"]),
            "n_channels": int(shape_meta["n_channels"]),
        })

        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_uri", required=True, help="例如 s3://omni-eeg-01/lance")
    parser.add_argument("--input_table", required=True, help="例如 eeg_foundation_train")
    parser.add_argument("--output", required=True, help="输出 token Lance dataset 路径")
    parser.add_argument("--electrode_vocab", required=True, help="electrode_vocab.json 路径")

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_channels", type=int, default=26)
    parser.add_argument("--seq_len", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0, help="调试用，只转换前 N 行；0 表示全部")
    parser.add_argument("--dataset_name", default="foundation_lance")
    parser.add_argument("--no_denorm", action="store_true", help="不把 signals 乘回 200")

    parser.add_argument("--aws_access_key_id", default="AKLTYmE3ODQyYWQxNWY3NDU1MThiOWY4NWVhMzdmNjAwYjQ")
    parser.add_argument("--aws_secret_access_key", default="TWpjNVptWTFaRGN3WkRjeE5HWTFNemt4WWpKa01URm1ORGcyWW1KbU1UQQ==")
    parser.add_argument("--endpoint", default="https://omni-eeg-01.tos-s3-cn-beijing.ivolces.com")
    parser.add_argument("--region", default="cn-beijing")
    parser.add_argument("--virtual_hosted_style_request", default="true")

    args = parser.parse_args()

    cfg = Config()
    id2name = load_id2name(args.electrode_vocab)

    storage_options = {
        "aws_access_key_id": args.aws_access_key_id,
        "aws_secret_access_key": args.aws_secret_access_key,
        "endpoint": args.endpoint,
        "region": args.region,
        "virtual_hosted_style_request": args.virtual_hosted_style_request,
    }

    print("===== Opening Input Lance Table =====")
    db = lancedb.connect(uri=args.input_uri, storage_options=storage_options)
    tbl = db.open_table(args.input_table)
    ds = tbl.to_lance()

    print(ds.schema)
    print("fields:", ds.schema.names)
    total_rows = ds.count_rows()
    if args.limit and args.limit > 0:
        total_rows = min(total_rows, args.limit)
    print("rows to convert:", total_rows)

    read_columns = [
        "signals",
        "electrode_ids",
        "sample_id",
        "subject_id",
        "edf_relpath",
        "segment_start_sec",
        "channel_counts",
        "global_idx",
    ]
    read_columns = [c for c in read_columns if c in ds.schema.names]

    schema = None
    shape_meta = None
    mode = "overwrite"
    written = 0

    for start in range(0, total_rows, args.batch_size):
        end = min(start + args.batch_size, total_rows)
        indices = list(range(start, end))

        batch = ds.take(indices=indices, columns=read_columns)

        raw_samples, raw_meta = foundation_batch_to_raw_samples(
            batch,
            id2name=id2name,
            n_channels=args.n_channels,
            seq_len=args.seq_len,
            dataset_name=args.dataset_name,
            denorm_clip_divide_200=not args.no_denorm,
        )

        processed_samples = preprocess_all_samples(raw_samples, cfg)

        if len(processed_samples) != len(raw_samples):
            raise ValueError(
                f"processed_samples length mismatch in batch {start}:{end}: "
                f"{len(processed_samples)} vs {len(raw_samples)}"
            )

        if schema is None:
            schema, shape_meta = make_schema_from_first_processed(processed_samples[0])
            print("===== Output Token Schema =====")
            print(schema)
            print("shape_meta:", shape_meta)

        rows = processed_to_rows(processed_samples, raw_meta, shape_meta)
        table = pa.Table.from_pylist(rows, schema=schema)

        lance.write_dataset(
            table,
            args.output,
            mode=mode,
        )
        mode = "append"
        written += len(rows)
        print(f"written {written}/{total_rows}", flush=True)

    print("Done.")
    print(f"Wrote {written} samples to {args.output}")


if __name__ == "__main__":
    main()