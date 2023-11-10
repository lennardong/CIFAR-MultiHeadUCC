from dataclasses import dataclass
import torch
from typing import Dict
import numpy as np
import os
import lightning as L
import torch.nn.functional as F

# define modules
from model import UCCModel

############################################
# DATA - PYTORCH DATALOADER
############################################

from dataloader import create_dataloaders
from torchvision.transforms import functional as TF
import random

# Inits

@dataclass
class DataConfig:
    npz_path_: str = '../data/Splitted CIFAR10.npz'
    lower_ucc: int = 2
    upper_ucc: int = 4
    bag_size: int = 300
    bag_fraction: float = 0.3
    batch_size: int = 32
    transform: Dict = None

transforms = {
    'random_horizontal_flip': lambda img: TF.hflip(img) if random.random() > 0.5 else img,
    'random_vertical_flip': lambda img: TF.vflip(img) if random.random() > 0.5 else img,
    'color_jitter': lambda img: TF.adjust_brightness(img, brightness_factor=random.uniform(0.8, 1.2)),
    'normalize': lambda img: TF.normalize(img, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
}

# Instance

DATA_CONFIG_1 = DataConfig(
    lower_ucc=1,
    upper_ucc=4,
    bag_size=50,
    bag_fraction=0.3,
    batch_size=10,
    transform=transforms)

dataloaders = create_dataloaders(**DATA_CONFIG_1.__dict__)
train_loader = dataloaders['train']
val_loader = dataloaders['val']

#
# # Testing the dataloaders
# for images, labels in dataloaders['train']:
#     print(f'Images batch shape: {images.shape}')
#     print(f'Labels batch shape: {labels.shape}')
#     print(f'Labels: {labels}')
#     print(f'Images: {images}')
#     break


############################################
# MODEL - PYTORCH MODEL
############################################

# Configs
@dataclass
class ModelConfig:
    num_bins: int = 10
    sigma: float = 0.1
    dropout_rate: float = 0.1
    num_classes: int = 4
    embedding_size: int = 110
    fc2_size: int = 512


# Load base model
model_config_test = ModelConfig()
model = UCCModel(**model_config_test.__dict__)

############################################
# DATA - PYTORCH DATALOADER
############################################


def loss_multihead(logits, decoded_img, labels, original_imgs, ucc_loss_weight=0.5):

    ae_loss_weight = 1 - ucc_loss_weight

    ucc_loss = F.cross_entropy(logits, labels)
    ae_loss = F.mse_loss(decoded_img, original_imgs)
    combined_loss = (ucc_loss_weight * ucc_loss) + (ae_loss_weight * ae_loss)

    return ucc_loss, ae_loss, combined_loss

############################################
# LIGHTNING MODULE
############################################


class UCCModule(L.LightningModule):
    def __init__(self, ucc_model, loss_function):
        super().__init__()

        # Modules
        self.ucc_model = ucc_model
        self.loss_function = loss_function

    def model_outputs(self, x):
        # get channel shapes
        batch_size, num_instances, num_channel, height, width = x.shape
        x_flat = x.view(-1, num_channel, height, width)

        ############################
        # Forward
        ############################

        # Encoder
        embeddings = self.ucc_model.encoder(x_flat)

        # Head1: Decoder
        decoded_img_flat = self.ucc_model.decoder(embeddings)
        decoded_img = decoded_img_flat.view(batch_size, num_instances, num_channel, height,
                                            width)  # Shape: (batch_size, num_instances, num_channel, height, width)

        # Head2: KDE

        embeddings_reshaped = embeddings.view(batch_size, num_instances, embeddings.shape[
            -1])  # Shape: (batch_size, num_instances, embedding_size)
        feature_distribution = self.ucc_model.kde(embeddings_reshaped, self.ucc_model.num_bins,
                                                  self.ucc_model.sigma)  # Shape: (batch_size, num_bins * embedding_size)
        logits = self.ucc_model.mlp_classifier(feature_distribution)

        return logits, decoded_img


    def training_step(self, batch, batch_idx):
        x, y = batch

        logits, decoded_img = self.model_outputs(x)
        ucc_loss, ae_loss, train_loss = self.loss_function(logits, decoded_img, y, x)

        self.log_dict({"train_loss": train_loss, "ucc_loss": ucc_loss, "ae_loss": ae_loss})

        return train_loss

    def validation_step(self, batch, batch_idx):
        """
        Each validation_step processes a single batch, and the results are automatically aggregated over all batches.
        """
        x, y = batch
        logits, decoded_img = self.model_outputs(x)

        # compute loss
        ucc_loss, ae_loss, val_loss = self.loss_function(logits, decoded_img, y, x)

        # Compute accuracy
        _, preds = torch.max(logits, dim=1)
        val_acc = (preds == y).float().mean().item()

        # logging
        self.log_dict({
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        return {'val_loss': val_loss, 'val_acc': val_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

ucc_module = UCCModule(model, loss_multihead)

# Train the model
trainer = L.Trainer(max_steps=1000)
trainer.fit(model=ucc_module, train_dataloaders=train_loader, val_dataloaders= val_loader)
