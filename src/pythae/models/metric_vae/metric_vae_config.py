from pydantic.dataclasses import dataclass
import torch
from ..vae import VAEConfig
import pandas as pd


# @dataclass
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
    n_out_channels: int = 16  # number of layers to convolutional kernel
    distance_metric: str = "cosine"
    class_key_path: str = ''
    # class_key = None
    class_ignorance_flag: bool = False
    time_ignorance_flag: bool = False  # if true, we squeeze class info out of the nuisance partition
    time_similarity_threshold: float = 3  # specifies how close in age different observations need to be to be counted as positive pairs
    gamma: float = 1.0

    def __init__(self, class_key_path=None, **kwargs):
        self.__dict__.update(kwargs)

        self.class_key_path = class_key_path
        self.class_key = None
        if self.class_key_path is not None:
            self.load_dataset()

    def load_dataset(self):
        """
        Load the dataset from the specified file path using pandas.
        """
        class_key = pd.read_csv(self.class_key_path)

        if self.class_ignorance_flag & self.time_ignorance_flag:
            class_key = class_key.loc[:, ["snip_id", "predicted_stage_hpf", "perturbation_id"]]
        elif self.class_ignorance_flag:
            class_key = class_key.loc[:, ["snip_id", "perturbation_id"]]
        elif self.time_ignorance_flag:
            class_key = class_key.loc[:, ["snip_id", "predicted_stage_hpf"]]
        else:
            class_key = None

        self.class_key = class_key






