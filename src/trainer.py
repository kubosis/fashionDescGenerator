import copy

import torch
import torch.nn as nn
import torchmetrics
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from src.model import ImageCaptionModel


def test(model: ImageCaptionModel,
         data_loader: torch.utils.data.DataLoader,
         device: str,
         num_classes: int,
         epoch=0,
         total_epochs=0,
         pad_idx: int = 0):
    # Ignore the padding index in loss and accuracy calculations
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=pad_idx).to(device)

    model.eval()
    acc.reset()
    total_loss = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"Eval Epoch {epoch + 1}/{total_epochs}")

        for batch in pbar:
            images, captions = batch
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions, pad_index=pad_idx)
            # Output shape: (Batch, Seq_Len-1, Vocab)

            # Targets are the next words (shifted by 1)
            targets = captions[:, 1:]
            # Target shape: (Batch, Seq_Len-1)

            # Flatten for Loss/Metric calculation
            # Reshape to (Batch * (Seq_Len-1), Vocab)
            outputs_flat = outputs.reshape(-1, num_classes)
            # Reshape to (Batch * (Seq_Len-1))
            targets_flat = targets.reshape(-1)

            # Update metrics
            loss = loss_fn(outputs_flat, targets_flat)
            total_loss += loss.item()

            acc.update(outputs_flat, targets_flat)

    avg_loss = total_loss / len(data_loader)
    return acc.compute(), avg_loss


def train(model: ImageCaptionModel,
          optim: torch.optim.Optimizer,
          num_classes: int,
          train_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          scheduler: torch.optim.lr_scheduler.LRScheduler=None,
          save_path: str = "best_model.pth",
          device: str = 'cpu',
          epochs: int = 50,
          clip_norm: float = 1.0,
          patience: int = 5,
          min_delta: float = 0.0):
    scaler = GradScaler('cuda')

    pad_idx = vocab.pad

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=pad_idx).to(device)

    best_model_state = None
    best_test_loss = float('inf')

    # Early Stopping Counter
    epochs_no_improve = 0

    print(f"Starting training on {device} with Early Stopping (Patience={patience})...")
    model = model.to(device)

    collected_values = {
        "train_loss": [float("inf")],
        "test_loss": [float("inf")],
        "train_acc": [0],
        "test_acc": [0],
        "lr": [0],
        "epoch": [0]
    }

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        current_lr = optim.param_groups[0]['lr']

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")

        for batch in pbar:
            images, captions = batch
            images = images.to(device)
            captions = captions.to(device)

            optim.zero_grad()

            with autocast('cuda'):
                outputs = model(images, captions, pad_index=pad_idx)
                targets = captions[:, 1:]

                outputs_flat = outputs.reshape(-1, num_classes)
                targets_flat = targets.reshape(-1)

                loss = loss_fn(outputs_flat, targets_flat)

            # Scale loss and backprop
            scaler.scale(loss).backward()

            # Unscale for gradient clipping
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            # Step with scaler
            scaler.step(optim)
            scaler.update()

            total_train_loss += loss.item()
            accuracy.update(outputs_flat, targets_flat)

            if scheduler:
                scheduler.step()

            # 2. Update the progress bar text
            current_lr = optim.param_groups[0]['lr']

            # We use .set_postfix() to display the changing metrics
            pbar.set_postfix({
                "epoch": epoch + 1,
                "lr": f"{current_lr:.2e}",
                "loss": f"{loss.item():.4f}"
            })

        train_acc = accuracy.compute()
        test_acc, test_loss = test(model, test_loader, device, num_classes, pad_idx=pad_idx)

        avg_train_loss = total_train_loss / len(train_loader)

        collected_values["train_loss"].append(avg_train_loss)
        collected_values["test_loss"].append(test_loss)
        collected_values["test_acc"].append(test_acc.item())
        collected_values["train_acc"].append(train_acc.item())
        collected_values["epoch"].append(epoch+1)
        collected_values["lr"].append(current_lr)

        # --- EARLY STOPPING LOGIC ---
        # Check if result is better than best + minimum delta
        if test_loss < (best_test_loss - min_delta):
            best_test_loss = test_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, save_path)

            epochs_no_improve = 0  # Reset counter
            saved_msg = "-> Saved! (New Best)"
        else:
            epochs_no_improve += 1
            saved_msg = f"| No Improv ({epochs_no_improve}/{patience})"

        # Formatting output
        out_epoch = f"{epoch + 1:02d}"
        print(f"[EPOCH {out_epoch}/{epochs}] lr={current_lr:.2e} | "
              f"Train Loss: {avg_train_loss:.3f}, Acc: {train_acc:.3f} | "
              f"Test Loss: {test_loss:.3f}, Acc: {test_acc:.3f} {saved_msg}")

        # Reset train accuracy for next epoch
        accuracy.reset()

        # Trigger Stop
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered! Test accuracy did not improve for {patience} epochs.")
            break

    # Load best weights back into model before returning
    if best_model_state is not None:
        print(f"Restoring best model with Test Loss: {best_test_loss:.3f}")
        model.load_state_dict(best_model_state)

    return model, collected_values


if __name__ == '__main__':
    from src.dataset.utils import extract_vocab_from_h5py
    from src.dataset.dataset import ImageCaptionDataset
    from src.model import ImageCaptionModel, get_timm_cnn_pretrained_cnf

    H5_PATH = "./fashiongen_data/fashiongen_256_256_train.h5"
    H5_PATH_TEST = "./fashiongen_data/fashiongen_256_256_validation.h5"

    # 1. Extract vocab (requires opening file, but we close it after)
    vocab = extract_vocab_from_h5py(H5_PATH)
    print("Vocab extracted")

    # 2. Initialize Model
    model = ImageCaptionModel(len(vocab), embed_dim=1024, hidden_dim=512, num_hidden_layers=2)
    print("Model initialized")

    # 3. Create Dataloaders (Pass PATHS, not objects)
    # We pass 'model.cnn' so the dataset can read the normalization config
    train_loader = ImageCaptionDataset(H5_PATH, model.cnn, vocab).get_dataloader(
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    test_loader = ImageCaptionDataset(H5_PATH_TEST, model.cnn, vocab).get_dataloader(
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    print("Dataloaders created")

    optim = torch.optim.Adam(model.parameters(), lr=0.005)

    epochs = 10
    warmup_steps = int(0.1 * epochs * len(train_loader))
    main_steps = epochs * len(train_loader) - warmup_steps

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=main_steps)
        ],
        milestones=[warmup_steps]
    )

    print("Starting Training")
    # Note: 'len(vocab)' is passed as num_classes
    model, collected_values = train(model, optim, len(vocab), train_loader, test_loader, epochs=10, device='cuda',
                                    patience=2, scheduler=scheduler)

    import json

    with open("collected.json", "w") as f:
        json.dump(collected_values, f)
