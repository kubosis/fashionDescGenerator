from typing import Sequence, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision
from tqdm import tqdm
import timm

from src.dataset.utils import AutoVocab


class ImageCaptionDataset(Dataset):
    def __init__(self,
                 images: Sequence[str],
                 captions: Sequence[str],
                 cnn: torch.nn.Module,
                 vocab: AutoVocab,
                 transform=None,
                 preload: bool = False):
        """
        preload: If True, loads all images into RAM during initialization.
                 WARNING: Requires significant RAM for large datasets.
        """
        assert len(images) == len(captions)

        self.captions = captions
        self.vocab = vocab
        self.transform = transform

        # --- SETUP PREPROCESSING ---
        data_config = timm.data.resolve_model_data_config(cnn)
        self.base_preprocessing = timm.data.create_transform(**data_config, is_training=True)

        # --- PRELOAD LOGIC ---
        if preload:
            print(f"Preloading {len(images)} images into RAM...")
            self.images = []
            for img_path in tqdm(images):
                img_tensor = torchvision.io.decode_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
                self.images.append(img_tensor)
        else:
            self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_source = self.images[item]

        # 1. Load Image (if not preloaded)
        if isinstance(img_source, np.typing.ArrayLike):
            img_source = torch.tensor(img_source)
        else:
            img_source = torchvision.io.decode_image(img_source, mode=torchvision.io.ImageReadMode.RGB)

        # 2. Base Preprocessing
        image = self.base_preprocessing(img_source.permute(2, 0, 1).to(torch.float32))

        # 3. Augmentations (optional)
        if self.transform:
            image = self.transform(image)

        # 4. Text
        sentence_ids = torch.tensor(self.vocab.to_indices(self.captions[item][0]), dtype=torch.long)

        return torch.Tensor(image), sentence_ids

    def get_dataloader(self, batch_size, shuffle=False, num_workers=0):
        from functools import partial

        pad_idx = self.vocab.pad if hasattr(self.vocab, 'pad') else self.vocab.to_index("<PAD>")
        collate_wrapper = partial(collate_fn, pad_idx=pad_idx)

        return DataLoader(self,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          collate_fn=collate_wrapper,
                          num_workers=num_workers)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_idx: int):
    images, word_ids = zip(*batch)
    images = torch.stack(images, dim=0)
    word_ids_padded = torch.nn.utils.rnn.pad_sequence(
        word_ids,
        batch_first=True,
        padding_value=pad_idx
    )
    return images, word_ids_padded
