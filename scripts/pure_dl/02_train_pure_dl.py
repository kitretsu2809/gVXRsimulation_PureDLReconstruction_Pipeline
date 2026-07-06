from __future__ import annotations

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
    torch, nn, _ = _import_torch_or_exit()
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
            noisy_sinogram = torch.from_numpy(self.input_sinograms[sample_idx][None, :, :])
            target_sinogram = torch.from_numpy(self.target_sinograms[sample_idx][None, :, :])
            target_image = torch.from_numpy(self.target_images[sample_idx][None, :, :])
            
            # Interpolate the sparse sinogram to the dense shape so the network can repair it
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
    parser.add_argument("--batch-size", type=int, default=2) # Dual-domain uses more memory
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dynamically adjust default paths based on the scan method if they weren't overridden
    dataset_path = resolve_repo_path(args.dataset_path)
    if dataset_path.is_file() and dataset_path.name == "sparse_sinogram_dataset.npz":
        dataset_path = dataset_path.parent / f"sparse_sinogram_dataset_{args.scan_method}.npz"
        
    output_dir = Path(args.output_dir)
    if output_dir.name == "pure_dl_training":
        output_dir = output_dir.parent / f"pure_dl_training_{args.scan_method}"

    # Handle single file vs directory of batch npz files
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
            metadata = meta # Use first file's metadata for network architecture sizing
        # Determine target image size from metadata if available, otherwise infer from data
        if observed_target_size is None:
            if meta is not None and hasattr(meta, "image_size") and meta.image_size is not None:
                observed_target_size = meta.image_size
            else:
                # target_images shape: (N, H, W) or (N, 1, H, W) depending on loader; handle both
                if target_images.ndim == 4:
                    observed_target_size = target_images.shape[-1]
                elif target_images.ndim == 3:
                    observed_target_size = target_images.shape[-1]
                else:
                    raise ValueError("Unable to infer target image size from dataset")
            
        train_indices, val_indices = split_indices(len(input_sinograms), args.val_fraction, args.seed)
        train_datasets.append(DualDomainDataset(input_sinograms, target_sinograms, target_images, train_indices))
        val_datasets.append(DualDomainDataset(input_sinograms, target_sinograms, target_images, val_indices))

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Initializing PureDLPipeline...")
    # Get image size from metadata (assuming square reconstruction grid) or fall back to observed size
    if metadata is not None and hasattr(metadata, "image_size") and metadata.image_size is not None:
        target_size = metadata.image_size
    elif observed_target_size is not None:
        target_size = observed_target_size
    else:
        raise ValueError("image_size is not available in metadata and could not be inferred from data")
    model = PureDLPipeline(target_image_size=target_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    l1_loss = nn.L1Loss()

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
        "learning_rate": args.learning_rate,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "device": str(device),
        "metadata": metadata.__dict__,
        "epoch_logs": [],
    }

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for noisy_sino, target_sino, target_image in train_loader:
            noisy_sino = noisy_sino.to(device)
            target_sino = target_sino.to(device)
            target_image = target_image.to(device)

            final_image, clean_sinogram, rough_image = model(noisy_sino)
            
            loss_sino = l1_loss(clean_sinogram, target_sino)
            loss_rough_image = l1_loss(rough_image, target_image)
            loss_final_image = l1_loss(final_image, target_image)
            total_loss = loss_sino + loss_rough_image + loss_final_image

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            train_losses.append(float(total_loss.item()))

        model.eval()
        val_losses = []
        val_psnr_scores = []
        with torch.no_grad():
            for noisy_sino, target_sino, target_image in val_loader:
                noisy_sino = noisy_sino.to(device)
                target_sino = target_sino.to(device)
                target_image = target_image.to(device)

                final_image, clean_sinogram, rough_image = model(noisy_sino)
                
                loss_sino = l1_loss(clean_sinogram, target_sino)
                loss_rough_image = l1_loss(rough_image, target_image)
                loss_final_image = l1_loss(final_image, target_image)
                total_loss = loss_sino + loss_rough_image + loss_final_image
                
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
            }
        )
        save_history(history, output_dir)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
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
                    "metadata": metadata.__dict__,
                    "epoch": epoch,
                    "history": history,
                },
                best_checkpoint,
            )

        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_psnr={val_psnr:.3f}dB"
        )


if __name__ == "__main__":
    main()
