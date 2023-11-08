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
        orth_flag (bool): indicates whether or not to impose orthogonality constraint on latent dimensions
        gamma (float): weight factor that controls weight of orthogonality cost relative to rest of loss function
    """
    temperature: float = 1.0
    zn_frac: float = 0.1
    orth_flag: bool = False
    n_conv_layers: int = 5  # number of convolutional layers
    n_out_channels: int = 16 # number of layers to convolutional kernel
    distance_metric: str = "cosine"
    class_key: float = None
    class_ignorance_flag: bool = False # if true, we squeeze class info out of the nuisance partition
    # gamma: float = 1.0


