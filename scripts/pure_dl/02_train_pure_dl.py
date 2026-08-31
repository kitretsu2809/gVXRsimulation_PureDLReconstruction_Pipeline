from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ct_recon.paths import OUTPUTS_DIR, resolve_repo_path
from ct_recon.sparse_ct_reconstruction import _import_torch_or_exit, load_sparse_dataset, psnr_np, save_history
from ct_recon.pure_dl_net import PureDLPipeline

def compute_sobel_loss(pred, target, torch_F):
    kx = pred.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    ky = pred.new_tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)

    pred_gx = torch_F.conv2d(pred, kx, padding=1)
    pred_gy = torch_F.conv2d(pred, ky, padding=1)
    target_gx = torch_F.conv2d(target, kx, padding=1)
    target_gy = torch_F.conv2d(target, ky, padding=1)

    return torch_F.l1_loss(pred_gx, target_gx) + torch_F.l1_loss(pred_gy, target_gy)

def split_indices(count: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, int(count * val_fraction))
    val_indices = set(indices[:val_count])
    train_indices = [idx for idx in indices if idx not in val_indices]
    valid_indices = [idx for idx in indices if idx in val_indices]
    return train_indices, valid_indices

def main():
    torch, nn, F = _import_torch_or_exit()
    from torch.utils.data import DataLoader, Dataset, ConcatDataset

    class DualDomainDataset(Dataset):
        def __init__(self, input_sinograms, target_sinograms, target_images, indices):
            self.input_sinograms = input_sinograms
            self.target_sinograms = target_sinograms
            self.target_images = target_images
            self.indices = list(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):
            import torch.nn.functional as F
            
            sample_idx = self.indices[index]
            noisy_sinogram  = torch.from_numpy(
                np.ascontiguousarray(self.input_sinograms[sample_idx],  dtype=np.float32)
            ).unsqueeze(0)
            target_sinogram = torch.from_numpy(
                np.ascontiguousarray(self.target_sinograms[sample_idx], dtype=np.float32)
            ).unsqueeze(0)
            target_image    = torch.from_numpy(
                np.ascontiguousarray(self.target_images[sample_idx],    dtype=np.float32)
            ).unsqueeze(0)
            
            if noisy_sinogram.shape != target_sinogram.shape:
                noisy_sinogram = F.interpolate(
                    noisy_sinogram.unsqueeze(0), 
                    size=target_sinogram.shape[1:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
            return noisy_sinogram, target_sinogram, target_image

    parser = argparse.ArgumentParser(description="Train the SOTA Pure DL Reconstructor.")
    parser.add_argument("--dataset-path", default=str(OUTPUTS_DIR / "sparse_sinogram_dataset.npz"), help="Path to a single .npz file or a directory containing multiple .npz files.")
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "pure_dl_training"))
    parser.add_argument("--scan-method", type=str, choices=['centered', 'offset'], default='centered', help="Scan geometry type to train on.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps (effective batch size = batch_size * accum_steps).")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate. Use a smaller value (e.g. 1e-4) when resuming.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Path to a previously saved .pt checkpoint to resume training from. "
                             "Example: outputs/pure_dl_training_centered/best_model_centered.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = resolve_repo_path(args.dataset_path)
    if dataset_path.is_file() and dataset_path.name == "sparse_sinogram_dataset.npz":
        dataset_path = dataset_path.parent / f"sparse_sinogram_dataset_{args.scan_method}.npz"
        
    output_dir = Path(args.output_dir)
    if output_dir.name == "pure_dl_training":
        output_dir = output_dir.parent / f"pure_dl_training_{args.scan_method}"

    npz_files = []
    if dataset_path.is_dir():
        npz_files = list(dataset_path.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {dataset_path}")
        print(f"Discovered {len(npz_files)} dataset files for Big Data training.")
    else:
        npz_files = [dataset_path]

    train_datasets = []
    val_datasets = []
    metadata = None
    observed_target_size = None

    for npz_file in npz_files:
        print(f"Loading {npz_file.name}...")
        input_sinograms, target_sinograms, target_images, meta = load_sparse_dataset(npz_file)
        if metadata is None:
            metadata = meta
        if observed_target_size is None:
            if target_images.ndim in (3, 4):
                observed_target_size = int(target_images.shape[-1])
            elif meta is not None and hasattr(meta, "image_size") and meta.image_size is not None:
                observed_target_size = meta.image_size
            
        train_indices, val_indices = split_indices(len(input_sinograms), args.val_fraction, args.seed)
        train_datasets.append(DualDomainDataset(input_sinograms, target_sinograms, target_images, train_indices))
        val_datasets.append(DualDomainDataset(input_sinograms, target_sinograms, target_images, val_indices))

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_bs = args.batch_size * args.accum_steps
    print(f"Device: {device} | Physical Batch Size: {args.batch_size} | Accumulation Steps: {args.accum_steps} (Effective Batch Size: {effective_bs})")
    
    target_size = observed_target_size if observed_target_size is not None else (metadata.image_size if (metadata and hasattr(metadata, "image_size")) else 256)
    print(f"Initializing PureDLPipeline for {target_size}x{target_size} reconstruction...")
    if metadata is not None and hasattr(metadata, "image_size"):
        metadata.image_size = target_size

    model = PureDLPipeline(target_image_size=target_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    l1_loss = nn.L1Loss()

    start_epoch = 1
    if args.resume_checkpoint is not None:
        ckpt_path = resolve_repo_path(args.resume_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        print(f"Loading checkpoint for resume: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Restored optimizer state from checkpoint.")
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("Restored scheduler state from checkpoint.")
        start_epoch = checkpoint.get("epoch", 0) + 1
        prev_history = checkpoint.get("history", {})
        prev_logs = prev_history.get("epoch_logs", [])
        if prev_logs:
            best_val_loss = min(log["val_loss"] for log in prev_logs)
            print(f"Restored best_val_loss from checkpoint: {best_val_loss:.6f}")
        print(f"Checkpoint was saved at epoch {checkpoint.get('epoch', '?')}. Starting from epoch {start_epoch}.")

    output_dir = resolve_repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / f"best_model_{args.scan_method}.pt"
    last_checkpoint = output_dir / f"last_model_{args.scan_method}.pt"

    history = {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "scan_method": args.scan_method,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "effective_batch_size": effective_bs,
        "learning_rate": args.learning_rate,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "device": str(device),
        "metadata": metadata.__dict__,
        "epoch_logs": [],
    }

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses = []
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (noisy_sino, target_sino, target_image) in enumerate(train_loader):
            noisy_sino = noisy_sino.to(device)
            target_sino = target_sino.to(device)
            target_image = target_image.to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                final_image, clean_sinogram, rough_image = model(noisy_sino)
                
                loss_sino = l1_loss(clean_sinogram, target_sino)
                loss_rough_image = l1_loss(rough_image, target_image)
                loss_final_image = l1_loss(final_image, target_image)
                loss_edge = compute_sobel_loss(final_image, target_image, F)
                
                raw_loss = 0.1 * loss_sino + 0.1 * loss_rough_image + 0.4 * loss_final_image + 0.4 * loss_edge
                loss_accum = raw_loss / args.accum_steps

            scaler.scale(loss_accum).backward()

            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_losses.append(float(raw_loss.item()))
            
            print(
                f"Epoch {epoch}/{args.epochs} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {raw_loss.item():.4f}",
                end="\r",
                flush=True,
            )

        model.eval()
        val_losses = []
        val_psnr_scores = []
        with torch.no_grad():
            for noisy_sino, target_sino, target_image in val_loader:
                noisy_sino = noisy_sino.to(device)
                target_sino = target_sino.to(device)
                target_image = target_image.to(device)

                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    final_image, clean_sinogram, rough_image = model(noisy_sino)
                    loss_sino = l1_loss(clean_sinogram, target_sino)
                    loss_rough_image = l1_loss(rough_image, target_image)
                    loss_final_image = l1_loss(final_image, target_image)
                    loss_edge = compute_sobel_loss(final_image, target_image, F)
                    total_loss = 0.1 * loss_sino + 0.1 * loss_rough_image + 0.4 * loss_final_image + 0.4 * loss_edge
                val_losses.append(float(total_loss.item()))


                predictions_np = final_image.detach().cpu().numpy()
                targets_np = target_image.detach().cpu().numpy()
                for pred, target in zip(predictions_np, targets_np):
                    val_psnr_scores.append(psnr_np(np.clip(pred, 0.0, 1.0), target))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        val_psnr = float(np.mean(val_psnr_scores)) if val_psnr_scores else float("nan")
        history["epoch_logs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "lr": optimizer.param_groups[0]['lr']
            }
        )
        save_history(history, output_dir)
        
        scheduler.step()

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metadata": metadata.__dict__,
                "epoch": epoch,
                "history": history,
            },
            last_checkpoint,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "metadata": metadata.__dict__,
                    "epoch": epoch,
                    "history": history,
                },
                best_checkpoint,
            )

        # Clear the carriage return line and print the final epoch status
        print(" " * 80, end="\r")
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | val_psnr={val_psnr:.3f}dB",
            flush=True,
        )


if __name__ == "__main__":
    main()
