"""
PSCDE sampling script.

Runs the diffusion model on a paired test set and saves generated outputs.
The data config defaults to configs/dataset/pscde/paired_test.yaml, which
expects a pair list at data/test.txt (relative to the project root).

Usage:
    python run_pscde_sampling.py --ckpt <path/to/checkpoint.ckpt> [options]

Options:
    --ckpt           Path to model checkpoint (.ckpt)  [required]
    --model_config   Path to model config yaml         [default: configs/model/cldm_v21_dynamic.yaml]
    --data_config    Path to dataset config yaml       [default: configs/dataset/pscde/paired_test.yaml]
    --output         Output directory                  [default: results/PSCDE]
    --steps          Diffusion sampling steps          [default: 50]
    --batch_size     Override batch size from config
    --device         cuda / cpu / auto                 [default: auto]
    --seed           Random seed                       [default: 42]
    --max_samples    Cap number of processed images
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Ensure the CDTSDE package root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils.common import instantiate_from_config, load_state_dict
from model.cldm import ControlLDM

# ── Default paths (overridable via CLI) ────────────────────────────────────────
MODEL_CONFIG = os.path.join(SCRIPT_DIR, "configs/model/cldm_v21_dynamic.yaml")
DATA_CONFIG  = os.path.join(SCRIPT_DIR, "configs/dataset/pscde/paired_test.yaml")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "results/PSCDE")
# ───────────────────────────────────────────────────────────────────────────────


def check_device(device_str: str) -> str:
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device_str = "cpu"
    return device_str


@torch.no_grad()
def run_batch(model: ControlLDM, batch: dict, steps: int, device: str) -> dict:
    """Run one batch through the model and return result tensors."""
    z, c = model.get_input(batch, model.first_stage_key, bs=batch["jpg"].shape[0])

    cond_dict = {
        "c_concat":   [c["c_concat"][0]],
        "c_crossattn":[c["c_crossattn"][0]],
        "c_latent":   [c["c_latent"][0]],
    }
    samples = model.sample_log(cond=cond_dict, steps=steps)

    return {
        "generated": samples,                        # [0,1]
        "source":    batch["hint"],                  # [0,1]
        "target":    (batch["jpg"] + 1.0) / 2.0,    # [0,1]
    }


def save_batch(results: dict, output_dir: str, start_idx: int) -> None:
    gen_dir = os.path.join(output_dir, "generated")
    src_dir = os.path.join(output_dir, "source")
    tgt_dir = os.path.join(output_dir, "target")
    for d in (gen_dir, src_dir, tgt_dir):
        os.makedirs(d, exist_ok=True)

    n = results["generated"].shape[0]
    for i in range(n):
        idx = start_idx + i

        def to_uint8(t):
            arr = t[i].detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):   # CHW → HWC
                arr = arr.transpose(1, 2, 0)
            return (arr * 255).clip(0, 255).astype(np.uint8)

        Image.fromarray(to_uint8(results["generated"])).save(
            os.path.join(gen_dir, f"{idx:04d}.png"))
        Image.fromarray(to_uint8(results["source"])).save(
            os.path.join(src_dir, f"{idx:04d}.png"))
        Image.fromarray(to_uint8(results["target"])).save(
            os.path.join(tgt_dir, f"{idx:04d}.png"))

    print(f"[INFO] Saved images {start_idx:04d} – {start_idx + n - 1:04d}")


def parse_args():
    parser = argparse.ArgumentParser(description="PSCDE sampling")
    parser.add_argument("--ckpt",         type=str, required=True,
                        help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--model_config", type=str, default=MODEL_CONFIG,
                        help=f"Path to model config yaml (default: {MODEL_CONFIG})")
    parser.add_argument("--data_config",  type=str, default=DATA_CONFIG,
                        help=f"Path to dataset config yaml (default: {DATA_CONFIG})")
    parser.add_argument("--output",       type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--steps",        type=int, default=50,
                        help="Diffusion sampling steps (default: 50)")
    parser.add_argument("--batch_size",   type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--device",       type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--max_samples",  type=int, default=None,
                        help="Cap the number of images processed")
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    device = check_device(args.device)

    # ── Validate paths ────────────────────────────────────────────────────────
    for label, path in [("Checkpoint", args.ckpt),
                        ("Model config", args.model_config),
                        ("Data config", args.data_config)]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found: {path}")
            sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading model config: {args.model_config}")
    model_cfg = OmegaConf.load(args.model_config)

    print("[INFO] Instantiating model …")
    model: ControlLDM = instantiate_from_config(model_cfg)

    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    state_dict = torch.load(args.ckpt, map_location=device)
    # Unwrap lightning checkpoint wrapper if present
    raw = state_dict.get("state_dict", state_dict)
    # Drop keys belonging to preprocess_model (not present in this codebase)
    filtered = {k: v for k, v in raw.items() if not k.startswith("preprocess_model.")}
    dropped = len(raw) - len(filtered)
    if dropped:
        print(f"[INFO] Dropped {dropped} preprocess_model keys not in this model")
    load_state_dict(model, filtered, strict=True)

    model.freeze()
    model.to(device)
    model.eval()
    print(f"[INFO] Model ready on {device}")

    # ── Build dataloader ──────────────────────────────────────────────────────
    print(f"[INFO] Loading data config: {args.data_config}")
    data_cfg = OmegaConf.load(args.data_config)

    dataset       = instantiate_from_config(data_cfg["dataset"])
    batch_transform = instantiate_from_config(data_cfg["batch_transform"])

    dl_cfg = dict(data_cfg["data_loader"])
    if args.batch_size is not None:
        dl_cfg["batch_size"] = args.batch_size
    dl_cfg["shuffle"]    = False
    dl_cfg["drop_last"]  = False

    dataloader = DataLoader(dataset, **dl_cfg)
    print(f"[INFO] Dataset: {len(dataset)} images, {len(dataloader)} batches")

    # ── Inference loop ────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    total_processed = 0
    t0 = time.time()

    print(f"\n[INFO] Starting sampling  (steps={args.steps}, output={args.output})")

    for batch_idx, batch in enumerate(dataloader):
        if args.max_samples and total_processed >= args.max_samples:
            print(f"[INFO] Reached max_samples={args.max_samples}, stopping.")
            break

        print(f"\n[INFO] Batch {batch_idx + 1}/{len(dataloader)}")

        batch = batch_transform(batch)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        t_batch = time.time()
        results = run_batch(model, batch, steps=args.steps, device=device)
        print(f"[INFO] Batch done in {time.time() - t_batch:.1f}s")

        save_batch(results, args.output, total_processed)
        total_processed += results["generated"].shape[0]

    elapsed = time.time() - t0
    print(f"\n[INFO] Done — {total_processed} images in {elapsed:.1f}s "
          f"({elapsed / max(total_processed, 1):.2f}s/img)")
    print(f"[INFO] Results → {args.output}/")


if __name__ == "__main__":
    main()
