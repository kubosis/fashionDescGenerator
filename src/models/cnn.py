import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    Basic configurable CNN
    """
    def __init__(self, hidden_dim: int, hidden_layers: int, out_dim: int):
        """
        :param hidden_dim: hidden layers
        :param hidden_layers:
        :param out_dim:
        """
        super().__init__()
        raise NotImplementedError
