import torch
import torch.nn as nn


def absolute_positional_encoding(n_tokens: int, d_model: int, device: torch.device = torch.device("cpu")):
    """
    Generate positional encoding.
    :param n_tokens: number of tokens
    :param d_model: dimension of model
    :param device: device to store positional encoding
    :return: positional encoding
    """
    # Generate position along sequence
    pos = torch.arange(n_tokens, dtype=torch.float, device=device).reshape(1, -1, 1)
    # Generate dimension along embedding
    dim = torch.arange(d_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    # Compute phase
    phase = pos / (10000 ** (dim / d_model))
    # Compute positional encoding as described in "Attention is all you need"
    pe = torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))
    return pe


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self):
        """
        Absolute positional encoding.
        """
        super().__init__()

    def forward(self, x):
        """
        Generate positional encoding to input.
        :param x: input of shape (batch, n_tokens, d_model)
        :return: positional encoding of shape (1, n_tokens, d_model)
        """
        batch, n_tokens, d_model = x.shape
        pe = absolute_positional_encoding(n_tokens, d_model, device=x.device)
        return pe

def absolute_positional_encoding_2d(n_tokens_x: int, n_tokens_y: int, d_model: int, device: torch.device = torch.device("cpu")):
    """
    Generate positional encoding.
    :param n_tokens: number of tokens
    :param d_model: dimension of model
    :param device: device to store positional encoding
    :return: positional encoding
    """
    # Generate position along sequence
    pos_x = torch.arange(n_tokens_x, dtype=torch.float, device=device).reshape(1, -1, 1)
    pos_x = pos_x.repeat(1, 1, n_tokens_y)
    pos_y = torch.arange(n_tokens_y, dtype=torch.float, device=device).reshape(1, 1, -1)
    pos_y = pos_y.repeat(1, n_tokens_x, 1)
    # Generate dimension along embedding
    dim = torch.arange(d_model/2, dtype=torch.float, device=device).reshape(-1, 1, 1)
    # Compute phase
    phase_x = pos_x / (10000 ** (dim / d_model))
    phase_y = pos_y / (10000 ** (dim / d_model))
    # Compute positional encoding as described in "Image transformer"
    pe_x = torch.where(dim.long() % 2 == 0, torch.sin(phase_x), torch.cos(phase_x))
    pe_y = torch.where(dim.long() % 2 == 0, torch.sin(phase_y), torch.cos(phase_y))
    pe = torch.cat((pe_x, pe_y), dim=0)
    return pe


class AbsolutePositionalEncoding2D(nn.Module):
    def __init__(self):
        """
        Absolute positional encoding.
        """
        super().__init__()

    def forward(self, x):
        """
        Generate positional encoding to input.
        :param x: input of shape (batch, n_tokens, d_model)
        :return: positional encoding of shape (1, n_tokens, d_model)
        """
        batch, d_model, n_tokens_x, n_tokens_y = x.shape
        pe = absolute_positional_encoding_2d(n_tokens_x, n_tokens_y, d_model, device=x.device).repeat(batch, 1, 1, 1)
        return pe