import torch
import torch.nn as nn
import timm
from typing_extensions import Self


class ImageCaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_hidden_layers=2, model_name='tf_efficientnetv2_b0.in1k'):
        super().__init__()

        # 1. ENCODER
        self.cnn = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
        self.cnn_out_dim = self.cnn.num_features
        # Freeze Pretrained CNN Weights (Stop Backprop)
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.linear_h = nn.Linear(self.cnn_out_dim, hidden_dim * num_hidden_layers)
        self.linear_c = nn.Linear(self.cnn_out_dim, hidden_dim * num_hidden_layers)

        # 2. DECODER
        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_hidden_layers,
            batch_first=True,
            #dropout=0.5 if num_hidden_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.num_layers = num_hidden_layers
        self.hidden_dim = hidden_dim

        # Inference Cache
        self._h = None
        self._c = None

    def train(self, mode: bool = True) -> Self:
        """override default train so that cnn still stays in eval mode"""
        super().train(mode)
        self.cnn.eval()
        return self

    def forward(self, images, captions, pad_index=0):
        # 1. Image Features
        features = self.cnn(images)

        # 2. Init Hidden States
        h0 = self.linear_h(features).view(-1, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = self.linear_c(features).view(-1, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        # 3. Prepare Embeddings (Teacher Forcing: Input is everything except last token <EOS>)
        embeds = self.embed(captions[:, :-1])

        # We subtract 1 because we removed the <EOS> token from the input
        lengths = (captions != pad_index).sum(dim=1).cpu() - 1
        lengths = torch.clamp(lengths, min=1)

        # 4. Pack
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False
        )

        # 5. LSTM Pass
        packed_output, _ = self.lstm(packed, (h0, c0))
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # 7. Predict Logits
        outputs = self.fc(lstm_out)
        return outputs

    def predict_step(self, token, image=None):
        """
        Stateful prediction for one step at a time.

        :param token: Tensor of shape (1, 1) containing the index of the current input word.
        :param image: Tensor of shape (1, 3, H, W). Only required for the FIRST step of caption generation.
        """
        # If image is provided, we are starting a new sequence
        if image is not None:
            features = self.cnn(image)
            # Initialize state from image
            self._h = self.linear_h(features).view(-1, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
            self._c = self.linear_c(features).view(-1, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        # Ensure we have state (User must provide image on first call)
        assert self._h is not None, "Error: Call predict_step with 'image' first to initialize."

        # Embed the single token
        # token shape: (1, 1) -> embeds shape: (1, 1, embed_dim)
        embeds = self.embed(token)

        # Run LSTM for one step, cache hidden state and cell state
        output, (self._h, self._c) = self.lstm(embeds, (self._h, self._c))

        # Output shape: (1, 1, hidden_dim) -> (1, 1, vocab_size)
        pred = self.fc(output)

        # Return the logits for the next word
        return pred.squeeze(0).squeeze(0)  # (vocab_size)

def get_timm_cnn_pretrained_cnf(model: ImageCaptionModel) -> dict:
    """
    Timm models have specific normalization requirements.
    This extracts them automatically.

    :return: (dict) containing `mean`, `std` and `interpolation` of the pretrained model for \
        preprocessing
    """
    cnf = model.cnn.pretrained_cfg
    return cnf
