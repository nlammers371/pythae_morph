"""This module implements a slight variation on the base trainer class that tracks model performance metrics in detail
    over the course of training. Beyond this, some default options are also changed.

Available models:
------------------

.. autosummary::
    ~pythae.models.AE
    ~pythae.models.VAE
    ~pythae.models.BetaVAE
    ~pythae.models.DisentangledBetaVAE
    ~pythae.models.BetaTCVAE
    ~pythae.models.IWAE
    ~pythae.models.MSSSIM_VAE
    ~pythae.models.INFOVAE_MMD
    ~pythae.models.WAE_MMD
    ~pythae.models.VAMP
    ~pythae.models.SVAE
    ~pythae.models.VQVAE
    ~pythae.models.RAE_GP
    ~pythae.models.HVAE
    ~pythae.models.RHVAE
    :nosignatures:
"""

from .base_trainer_verbose import BaseTrainerVerbose
from .base_training_verbose_config import BaseTrainerVerboseConfig

__all__ = ["BaseTrainerVerbose", "BaseTrainerVerboseConfig"]
