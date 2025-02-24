""" Here are implemented the trainers used to train the Autoencoder models
"""

from .adversarial_trainer import AdversarialTrainer, AdversarialTrainerConfig
from .base_trainer import BaseTrainer, BaseTrainerConfig
from .base_trainer_verbose import BaseTrainerVerbose, BaseTrainerVerboseConfig
from .coupled_optimizer_adversarial_trainer import (
    CoupledOptimizerAdversarialTrainer,
    CoupledOptimizerAdversarialTrainerConfig,
)
from .coupled_optimizer_trainer import (
    CoupledOptimizerTrainer,
    CoupledOptimizerTrainerConfig,
)

__all__ = [
    "BaseTrainer",
    "BaseTrainerConfig",
    "BaseTrainerVerbose",
    "BaseTrainerVerboseConfig",
    "CoupledOptimizerTrainer",
    "CoupledOptimizerTrainerConfig",
    "AdversarialTrainer",
    "AdversarialTrainerConfig",
    "CoupledOptimizerAdversarialTrainer",
    "CoupledOptimizerAdversarialTrainerConfig",
]
