import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base import BaseAE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..nn.default_architectures import Encoder_VAE_MLP
from .metric_vae_config import MetricVAEConfig
class MetricVAE(BaseAE):
    """Vanilla Variational Autoencoder model.

    Args:
        model_config (VAEConfig): The Variational Autoencoder configuration setting the main
        parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: MetricVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "VAE"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' "
                    "where the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically"
                )

            encoder = Encoder_VAE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]

        # Check input to see if it is 5 dmensional, if so, then the model is being
        assert (len(x.shape) == 5) and (x.shape[1]==2)\
            , "Error with data shape. Missing contrastive pairs."

        x0 = torch.reshape(x[:, 0, :, :, :], (x.shape[0], x.shape[2], x.shape[3], x.shape[4])) # first set of images
        x1 = torch.reshape(x[:, 1, :, :, :], (x.shape[0], x.shape[2], x.shape[3], x.shape[4])) # second set with matched contrastive pairs

        encoder_output0 = self.encoder(x0)
        encoder_output1 = self.encoder(x1)

        mu0, log_var0 = encoder_output0.embedding, encoder_output0.log_covariance
        mu1, log_var1 = encoder_output1.embedding, encoder_output1.log_covariance

        std0 = torch.exp(0.5 * log_var0)
        std1 = torch.exp(0.5 * log_var1)

        z0, eps0 = self._sample_gauss(mu0, std0)
        recon_x0 = self.decoder(z0)["reconstruction"]

        z1, eps1 = self._sample_gauss(mu1, std1)
        recon_x1 = self.decoder(z1)["reconstruction"]

        # combine
        x_out = torch.cat([x0, x1], axis=0)
        recon_x_out = torch.cat([recon_x0, recon_x1], axis=0)
        mu_out = torch.cat([mu0, mu1], axis=0)
        log_var_out = torch.cat([log_var0, log_var1], axis=0)
        z_out = torch.cat([z0, z1], axis=0)

        loss, recon_loss, kld, nt_xent = self.loss_function(recon_x_out, x_out, mu_out, log_var_out, z_out)

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            contrastive_loss=nt_xent,
            loss=loss,
            recon_x=recon_x_out,
            z=z_out,
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, z):

        # calculate reconstruction error
        if self.model_config.reconstruction_loss == "mse":
            recon_loss = (
                0.5
                * F.mse_loss(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ).sum(dim=-1)
            )

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        # Calculate cross-entropy wrpt a standard multivariate Gaussian
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        nt_xent_loss = self.contrastive_loss(features=mu)

        return (recon_loss + KLD).mean(dim=0) + nt_xent_loss, recon_loss.mean(dim=0), KLD.mean(dim=0), nt_xent_loss

    def contrastive_loss(self, features, temperature=1, n_views=2):

        batch_size = int(features.shape[0] / n_views)

        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        # Normalize each latent vector. This simplifies the process of calculating cosie differences
        features = F.normalize(features, dim=1)

        # Due to above normalization, sim matrix entries are same as cosine differences
        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     n_views * batch_size, n_views * batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal, since this is the comparison of image with itself
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # Construct logits matrix with positive examples as firs column
        logits = torch.cat([positives, negatives], dim=1)

        # These labels tell the cross-entropy function that the positive example for each row is in the first column (col=0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # Apply temperature parameter
        logits = logits / temperature

        # initialize cross entropy loss
        loss_fun = torch.nn.CrossEntropyLoss()

        loss = loss_fun(logits, labels)

        return loss

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.

        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """

        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size

        log_p = []

        for i in range(len(data)):
            x = data[i].unsqueeze(0)

            log_p_x = []

            for j in range(n_full_batch):
                x_rep = torch.cat(batch_size * [x])

                encoder_output = self.encoder(x_rep)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance

                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)

                log_q_z_given_x = -0.5 * (
                    log_var + (z - mu) ** 2 / torch.exp(log_var)
                ).sum(dim=-1)
                log_p_z = -0.5 * (z ** 2).sum(dim=-1)

                recon_x = self.decoder(z)["reconstruction"]

                if self.model_config.reconstruction_loss == "mse":

                    log_p_x_given_z = -0.5 * F.mse_loss(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1) - torch.tensor(
                        [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                    ).to(
                        data.device
                    )  # decoding distribution is assumed unit variance  N(mu, I)

                elif self.model_config.reconstruction_loss == "bce":

                    log_p_x_given_z = -F.binary_cross_entropy(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1)

                log_p_x.append(
                    log_p_x_given_z + log_p_z - log_q_z_given_x
                )  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
        return np.mean(log_p)