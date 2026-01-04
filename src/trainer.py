import copy

import torch
import torch.nn as nn
import torchmetrics
from tqdm import tqdm

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
        pbar = tqdm(train_loader, desc=f"Eval Epoch {epoch + 1}/{total_epochs}")

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

    pad_idx = vocab.pad

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=pad_idx).to(device)

    best_model_state = None
    best_test_acc = 0.0

    # Early Stopping Counter
    epochs_no_improve = 0

    print(f"Starting training on {device} with Early Stopping (Patience={patience})...")

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

            outputs = model(images, captions, pad_index=pad_idx)
            targets = captions[:, 1:]

            outputs_flat = outputs.reshape(-1, num_classes)
            targets_flat = targets.reshape(-1)

            loss = loss_fn(outputs_flat, targets_flat)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optim.step()

            total_train_loss += loss.item()
            accuracy.update(outputs_flat, targets_flat)

            # 2. Update the progress bar text
            current_lr = optim.param_groups[0]['lr']

            # We use .set_postfix() to display the changing metrics
            pbar.set_postfix({
                "epoch": epoch + 1,
                "lr": f"{current_lr:.2e}",
                "loss": f"{loss.item():.4f}"
            })

        if scheduler:
            scheduler.step()

        train_acc = accuracy.compute()
        test_acc, test_loss = test(model, test_loader, device, num_classes, pad_idx=pad_idx)

        avg_train_loss = total_train_loss / len(train_loader)

        # --- EARLY STOPPING LOGIC ---
        # Check if result is better than best + minimum delta
        if test_acc > (best_test_acc + min_delta):
            best_test_acc = test_acc
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
        print(f"Restoring best model with Test Acc: {best_test_acc:.3f}")
        model.load_state_dict(best_model_state)

    return model


if __name__ == '__main__':
    import h5py
    from src.dataset.utils import extract_vocab_from_h5py
    from src.dataset.dataset import ImageCaptionDataset
    from src.model import ImageCaptionModel, get_timm_cnn_pretrained_cnf

    H5_PATH = "../fashiongen_data/fashiongen_256_256_train.h5"
    vocab = extract_vocab_from_h5py(H5_PATH)
    print("Vocab extracted")

    H5_PATH_TEST = "../fashiongen_data/fashiongen_256_256_validation.h5"
    with h5py.File(H5_PATH, "r") as f, h5py.File(H5_PATH, "r") as f2:
        images = f["input_image"]
        descriptions = f["input_description"]
        print("Train data loaded")

        images_test = f2["input_image"]
        descriptions_test = f2["input_description"]
        print("Test data loaded")

        model = ImageCaptionModel(len(vocab))
        cfg = get_timm_cnn_pretrained_cnf(model)
        print("Model initialized")

        train_loader = ImageCaptionDataset(images, descriptions, model.cnn, vocab, preload=False).get_dataloader(batch_size=1, shuffle=True)
        test_loader = ImageCaptionDataset(images_test, descriptions_test, model.cnn, vocab, preload=False).get_dataloader(batch_size=1, shuffle=False)
        print("Dataloaders created")

        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        print("Starting Training")
        train(model, optim, len(vocab), train_loader, test_loader)


