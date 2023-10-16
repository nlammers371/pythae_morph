from pydantic.dataclasses import dataclass
import torch
from ..vae import VAEConfig


@dataclass
class MetricVAEConfig(VAEConfig):
    r"""
    MetricVAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        temperature (float): Parameter dictating the temperature used in NT-Xent loss function. Default: 1
        zn_frac (float): fraction of latent dimensions to use for capturing nuisance variability
    """
    temperature: float = 1.0
    zn_frac: float = 0.1


