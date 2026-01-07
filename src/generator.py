""" this file is used to generate captions with pretrained model """

import argparse

import torch
import torchvision.io.image
import torchvision.transforms.v2 as v2

from src.dataset.utils import AutoVocab
from src.model import ImageCaptionModel, get_timm_cnn_pretrained_cnf

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str)
parser.add_argument("--image-path", type=str)
parser.add_argument("--vocab-path", type=str)


def generate_fast(model: ImageCaptionModel, image_path, vocab, max_len=100, device='cpu'):
    model.eval()

    image_tensor = torchvision.io.image.decode_image(image_path, mode="RGB")
    cfg = get_timm_cnn_pretrained_cnf(model)

    # use the same base transformation as during the training
    input_size = cfg["input_size"] # (C, H, W)
    target_h, target_w = input_size[1:]

    transformation = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(max(target_h, target_w), interpolation=v2.InterpolationMode(cfg["interpolation"])),
        v2.CenterCrop((target_h, target_w)),
        v2.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])
    image_tensor = transformation(image_tensor).unsqueeze(0).to(device)
    print(image_tensor)

    # Start with SOS
    next_token = torch.tensor([[vocab.to_index("<SOS>")]], device=image_tensor.device)

    generated_indices = []

    with torch.no_grad():
        logits = model.predict_step(next_token, image=image_tensor)
        for _ in range(max_len):
            # Pick next word (Greedy)
            next_token_id = logits.argmax().item()
            if next_token_id == vocab.to_index("<EOS>"):
                break
            generated_indices.append(next_token_id)
            next_token = torch.tensor([[next_token_id]], device=image_tensor.device)

            # This uses the cached self._h and self._c
            logits = model.predict_step(next_token, image=None)

    return [vocab.itt[idx] for idx in generated_indices]

def pretty_print_tokens(tokens: list[str]):
    is_after_dot = True
    tokens = [t for t in tokens if t not in {"<", ">", "br", "&", ";", "lt", "gt"}]
    for token in tokens:
        if not token in [".", ","]:
            print(" ", end="")

        if is_after_dot:
            print(token.capitalize(), end="")
        else:
            print(token, end="")

        if token == ".":
            is_after_dot = True
        else:
            is_after_dot = False
    print()

if __name__ == '__main__':
    args = parser.parse_args()

    vocab = AutoVocab.load(args.vocab_path)

    model = ImageCaptionModel(len(vocab), embed_dim=1024, hidden_dim=512, num_hidden_layers=2)
    model.load_state_dict(torch.load(args.model_path))

    tokens = generate_fast(model, args.image_path, vocab)
    pretty_print_tokens(tokens)


