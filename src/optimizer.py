"""
Hyperparameter Optimizer using Optuna.
Run this script to find the best architecture and training configuration.
"""

import argparse
import os
from pathlib import Path

import h5py
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset.dataset import ImageCaptionDataset
from src.dataset.utils import extract_vocab_from_h5py
from src.model import ImageCaptionModel
from src.trainer import test  # Reuse the test function

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_MODEL_PATH = "best_optuna_model.pth"

BEST_GLOBAL_LOSS = float('inf')


def objective(trial):
    global BEST_GLOBAL_LOSS

    # 1. Suggest Hyperparameters
    embed_dim = trial.suggest_categorical("embed_dim", [128, 256, 512, 1024])
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 3)

    # We test different backbones to see which feature extractor works best
    model_name = "tf_efficientnetv2_b0.in1k" # trial.suggest_categorical("model_name", ("tf_efficientnetv2_b0.in1k"))

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 5, 15)
    batch_size = 64

    # 2. Initialize Model
    vocab_size = len(vocab)
    model = ImageCaptionModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        model_name=model_name
    ).to(DEVICE)

    # 3. Initialize DataLoaders
    # Re-initialized per trial because transforms depend on the CNN backbone
    train_loader = ImageCaptionDataset(
        images_train_ref[:10000],
        desc_train_ref[:10000],
        model.cnn,
        vocab,
        preload=True
    ).get_dataloader(batch_size=batch_size, shuffle=True, num_workers=0)

    val_loader = ImageCaptionDataset(
        images_val_ref[:100],
        desc_val_ref[:100],
        model.cnn,
        vocab,
        preload=True
    ).get_dataloader(batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Setup Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad)

    # 5. Training Loop
    # We track best_trial_acc to return to Optuna
    best_trial_acc = 0.0
    pad_idx = vocab.pad

    print(f"\n--- Trial {trial.number}: Model={model_name}, LR={learning_rate:.1e} ---")

    for epoch in range(epochs):
        model.train()

        # Train Step
        for batch in tqdm(train_loader, desc=f"Ep {epoch + 1}/{epochs}", leave=False):
            images, captions = batch
            images, captions = images.to(DEVICE), captions.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images, captions, pad_index=pad_idx)
            targets = captions[:, 1:]

            loss = loss_fn(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation Step
        val_acc, val_loss = test(model, val_loader, DEVICE, vocab_size, epoch, epochs, pad_idx=pad_idx)
        val_acc = val_acc.item()

        print(f"Epoch {epoch + 1}: Val Acc={val_acc:.4f}")

        # --- KEY ADDITION: SAVE GLOBAL BEST MODEL ---
        if val_loss < BEST_GLOBAL_LOSS:
            BEST_GLOBAL_LOSS = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"!!! New Global Best Saved to {BEST_MODEL_PATH} (Acc: {val_acc:.4f}) !!!")

        # Report to Optuna
        trial.report(val_acc, epoch)

        # Handle Pruning
        if trial.should_prune():
            print("-> Trial Pruned!")
            raise optuna.TrialPruned()

        best_trial_acc = max(best_trial_acc, val_acc)

    return best_trial_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--name", type=str, default="fashion_study", help="Study name")
    parser.add_argument("--data-path", type=str, default="./fashiongen_data", help="path to data folder with the dataset")
    args = parser.parse_args()

    # 1. Global Data Loading
    data_path = Path(args.data_path)
    H5_PATH_TRAIN = data_path / "fashiongen_256_256_train.h5"
    H5_PATH_VAL = data_path / "fashiongen_256_256_validation.h5"
    BEST_MODEL_PATH = data_path / BEST_MODEL_PATH
    if not os.path.exists(H5_PATH_TRAIN):
        raise FileNotFoundError(f"Data not found at {H5_PATH_TRAIN}")

    vocab = extract_vocab_from_h5py(str(H5_PATH_TRAIN))

    # Open files globally for read access
    f_train = h5py.File(H5_PATH_TRAIN, "r")
    f_val = h5py.File(H5_PATH_VAL, "r")

    # Store references
    images_train_ref = f_train["input_image"]
    desc_train_ref = f_train["input_description"]
    images_val_ref = f_val["input_image"]
    desc_val_ref = f_val["input_description"]

    print(f"Data Loaded. Vocab Size: {len(vocab)}")

    # 2. Setup Optuna
    storage_url = f"sqlite:///{data_path}/{args.name}.db"
    study = optuna.create_study(
        study_name=args.name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )

    print(f"Starting Optimization: {args.trials} trials...")
    try:
        study.optimize(objective, n_trials=args.trials)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")

    # 3. Results
    print("\n" + "=" * 40)
    print(f"Global Best Loss Achieved: {BEST_GLOBAL_LOSS:.4f}")
    print(f"Best Model Saved to: {BEST_MODEL_PATH}")
    print("=" * 40)

    trial = study.best_trial
    print(f"Best Trial Value: {trial.value}")
    print("Best Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    f_train.close()
    f_val.close()