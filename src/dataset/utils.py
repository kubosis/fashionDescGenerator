import pickle
import os

import nltk
import h5py
from tqdm import tqdm

#nltk.download('punkt')
#nltk.download('punkt_tab')

class AutoVocab:
    def __init__(self):
        # token to index
        self.tti = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        # index to token
        self.itt = ["<PAD>", "UNK", "<SOS>", "<EOS>"]

    def __len__(self):
        return len(self.itt)

    @staticmethod
    def tokenizer_eng(text: str | bytes):
        """
        Standard NLTK tokenizer
        Only text cleaning: lowercase
        """
        if isinstance(text, (bytes, bytearray)):
            try:
                # Try standard UTF-8
                text = text.decode('utf-8')
            except UnicodeDecodeError:
                # Fallback to Latin-1
                text = text.decode('latin-1')
        return nltk.tokenize.word_tokenize(text.lower())

    @property
    def pad(self) -> int:
        """Get pad token index from the vocab"""
        return self.tti["<PAD>"]

    def add_words_from_sentence(self, sentence: str):
        word_list = self.tokenizer_eng(sentence)

        idx = len(self)
        for word in word_list:
            if word not in self.tti:
                self.tti[word] = idx
                self.itt.append(word)
                idx += 1

    def build_vocab(self, sentence_list: list[str]):
        for s in sentence_list:
            self.add_words_from_sentence(s)

    def to_index(self, word: str) -> int:
        word = word.lower()
        return self.tti[word] if word in self.tti else self.tti["<UNK>"]

    def to_indices(self, sentence: str | bytes) -> list[int]:
        word_list = self.tokenizer_eng(sentence)
        return [self.tti[word] if word in self.tti else self.tti["<UNK>"] for word in word_list]

    def __str__(self):
        return f"AutoVocab(len={len(self)})"

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

def extract_vocab_from_h5py(h5_path: str, save_path: str = "vocab.pkl") -> AutoVocab:
    # ensure nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

    if os.path.exists(save_path):
        print("Vocab already created, loading from cache...")
        # use cached object instead
        with open(save_path, "rb") as f:
            return pickle.load(f)

    vocab = AutoVocab()

    print(f"Opening {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        descriptions = f["input_description"]  # Shape (N, 1)
        total_samples = descriptions.shape[0]
        print(f"Found {total_samples} descriptions. Building vocab...")
        for i in tqdm(range(0, total_samples)):

            for row in descriptions[i]:
                vocab.add_words_from_sentence(row.decode("utf-8", errors="replace"))

        print(f"Vocabulary built! Total unique tokens: {len(vocab)}")

        # Save the vocab object to disk
        with open(save_path, "wb") as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary saved to {save_path}")

        return vocab
