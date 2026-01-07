from typing import Sequence, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as v2
import h5py

from src.dataset.utils import AutoVocab


class ImageCaptionDataset(Dataset):
    def __init__(self,
                 h5_path,
                 cnn: torch.nn.Module,
                 vocab: AutoVocab,
                 transform=None,
                 ):
        """
        h5_path: str, path to the .h5 file
        vocab: AutoVocab object
        cnn: The cnn model (to extract preprocessing config)
        """
        self.h5_path = h5_path
        self.vocab = vocab
        self.transform = transform

        # 2. Get dataset length (open briefly then close)
        with h5py.File(h5_path, "r") as f:
            self.length = len(f["input_image"])

        # 3. Placeholders for worker process
        self.h5_file = None
        self.images = None
        self.captions = None

        # --- SETUP PREPROCESSING ---
        cfg = cnn.pretrained_cfg

        # use the same base transformation as during the training
        input_size = cfg["input_size"]  # (C, H, W)
        target_h, target_w = input_size[1:]

        self.base_preprocessing = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(max(target_h, target_w), interpolation=v2.InterpolationMode(cfg["interpolation"])),
            v2.CenterCrop((target_h, target_w)),
            v2.Normalize(mean=cfg["mean"], std=cfg["std"]),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        # 4. Lazy Load: Open file ONLY inside the worker process
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
            self.images = self.h5_file["input_image"]
            self.captions = self.h5_file["input_description"]

        # 5. Load Data
        img_source = self.images[item]
        image_tensor = torch.from_numpy(img_source).permute(2, 0, 1).float() / 255.0

        # 6. Preprocessing
        # We ensure input is the correct shape/type for timm
        image = self.base_preprocessing(image_tensor)

        # 7. Text
        cap = self.captions[item][0]

        sentence_ids = torch.tensor(self.vocab.to_indices(cap), dtype=torch.long)

        return image, sentence_ids

    def get_dataloader(self, batch_size, shuffle=False, num_workers=0, **kwargs):
        from functools import partial

        pad_idx = self.vocab.pad if hasattr(self.vocab, 'pad') else self.vocab.to_index("<PAD>")
        collate_wrapper = partial(collate_fn, pad_idx=pad_idx)

        return DataLoader(self,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          collate_fn=collate_wrapper,
                          num_workers=num_workers,
                          **kwargs)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_idx: int):
    images, word_ids = zip(*batch)
    images = torch.stack(images, dim=0)
    word_ids_padded = torch.nn.utils.rnn.pad_sequence(
        word_ids,
        batch_first=True,
        padding_value=pad_idx
    )
    return images, word_ids_padded
