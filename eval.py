import argparse
import os
import json
import pickle
import glob
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms.v2 as v2
from PIL import Image

from src.dataset.utils import AutoVocab
from src.model import ImageCaptionModel, get_timm_cnn_pretrained_cnf

parser = argparse.ArgumentParser(description="Evaluate Image Captioning Model")
parser.add_argument("--model-path", type=str, required=True, help="Path to the .pth model checkpoint")
parser.add_argument("--vocab-path", type=str, required=True, help="Path to the vocab.pkl file")
parser.add_argument("--image-folder", type=str, required=True, help="Folder containing input images")
parser.add_argument("--stats-path", type=str, required=True, help="Path to the collected.json file")
parser.add_argument("--output-folder", type=str, default="eval_results", help="Folder to save results")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")


def tokens_to_string(tokens: list[str]) -> str:
    """Converts a list of tokens into a readable sentence."""
    clean_tokens = [t for t in tokens if t not in {"<SOS>", "<EOS>", "<PAD>", "<UNK>"}]
    sentence = " ".join(clean_tokens)
    # Basic cleanup
    sentence = sentence.replace(" .", ".").replace(" ,", ",")
    return sentence.capitalize()


def generate_caption(model: ImageCaptionModel, image_path: str, vocab: AutoVocab, device: str, max_len=25):
    model.eval()

    image_tensor = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB)

    cfg = get_timm_cnn_pretrained_cnf(model)
    input_size = cfg["input_size"]  # (C, H, W)
    target_h, target_w = input_size[1:]

    # Transform (matches training logic)
    transformation = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # Normalize 0-255 to 0-1
        v2.Resize(max(target_h, target_w), interpolation=v2.InterpolationMode(cfg["interpolation"])),
        v2.CenterCrop((target_h, target_w)),
        v2.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])

    input_tensor = transformation(image_tensor).unsqueeze(0).to(device)

    # Start with <SOS>
    next_token = torch.tensor([[vocab.to_index("<SOS>")]], device=device)
    generated_indices = []

    with torch.no_grad():
        # Initial step with image
        logits = model.predict_step(next_token, image=input_tensor)

        for _ in range(max_len):
            next_token_id = logits.argmax().item()

            if next_token_id == vocab.to_index("<EOS>"):
                break

            generated_indices.append(next_token_id)
            next_token = torch.tensor([[next_token_id]], device=device)

            # Subsequent steps (image=None)
            logits = model.predict_step(next_token, image=None)

    tokens = [vocab.itt[idx] for idx in generated_indices]
    return tokens_to_string(tokens)


def evaluate_images(args, model, vocab):
    print(f"Generating captions for images in {args.image_folder}...")

    # Find images (supports png, jpg, jpeg)
    exts = ['*.png', '*.jpg', '*.jpeg', '*.BMP']
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(args.image_folder, ext)))

    if not image_files:
        print("No images found in the specified folder.")
        return

    output_img_dir = os.path.join(args.output_folder, "generated_samples")
    os.makedirs(output_img_dir, exist_ok=True)

    for img_path in image_files:
        filename = os.path.basename(img_path)

        caption = generate_caption(model, img_path, vocab, args.device)

        # Plot
        plt.figure(figsize=(8, 8))

        pil_img = Image.open(img_path).convert("RGB")
        plt.imshow(pil_img)
        plt.axis('off')

        # Wrap text nicely
        import textwrap
        wrapped_caption = "\n".join(textwrap.wrap(caption, width=40))

        plt.title(f"Generated: {wrapped_caption}", fontsize=14, pad=20)

        save_path = os.path.join(output_img_dir, f"eval_{filename}")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")


# --- Task 2: Plot Statistics ---
def plot_statistics(args):
    print(f"Plotting statistics from {args.stats_path}...")

    if not os.path.exists(args.stats_path):
        print("Stats file not found!")
        return

    with open(args.stats_path, 'r') as f:
        data = json.load(f)

    output_stats_dir = os.path.join(args.output_folder, "training_plots")
    os.makedirs(output_stats_dir, exist_ok=True)

    epochs = data['epoch']
    train_loss = data['train_loss']
    test_loss = data['test_loss']
    train_acc = data['train_acc']
    test_acc = data['test_acc']
    lr = data['lr']

    # 1. Losses x Epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, test_loss, label='Test Loss', marker='o')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_stats_dir, "loss_history.png"))
    plt.close()

    # 2. Accuracy x Epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label='Train Acc', marker='o')
    plt.plot(epochs, test_acc, label='Test Acc', marker='o')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_stats_dir, "accuracy_history.png"))
    plt.close()

    # 3. Learning Rate x Epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lr, label='Learning Rate', color='orange', marker='x')
    plt.title('Learning Rate vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.grid(True)
    plt.savefig(os.path.join(output_stats_dir, "lr_history.png"))
    plt.close()

    print(f"Plots saved to {output_stats_dir}")


def main():
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    print("Loading vocab...")
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    print("Loading model...")
    # Using the dimensions from training script
    model = ImageCaptionModel(len(vocab), embed_dim=1024, hidden_dim=512, num_hidden_layers=2)

    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint)
    model.to(args.device)

    plot_statistics(args)
    evaluate_images(args, model, vocab)

    print("Evaluation Complete!")


if __name__ == "__main__":
    main()