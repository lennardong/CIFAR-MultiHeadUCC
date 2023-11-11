# DL libraries
import torch
import lightning as L
import numpy as np
import torch.nn.functional as F

# utils
import random

# models
from model import UCCModel, ModelConfig

# data
from torchvision.transforms import functional as TF
from dataloader import create_dataloaders, DataConfig


############################################
# LIGHTNING MODULE
############################################

class UCCModule(L.LightningModule):
    def __init__(self, ucc_model, learning_rate, ucc_weight = 0.5):
        super().__init__()

        # Modules
        self.ucc_model = ucc_model
        self.learning_rate = learning_rate
        self.ucc_weight = ucc_weight

    def forward_pass(self, x):
        # get channel shapes
        batch_size, num_instances, num_channel, height, width = x.shape
        x_flat = x.view(-1, num_channel, height, width)

        # Encoder
        embeddings_conv = self.ucc_model.encoder(x_flat)

        # Head1: Decoder
        decoded_img_flat = self.ucc_model.decoder(embeddings_conv)
        decoded_img = decoded_img_flat.view(batch_size, num_instances, num_channel, height,
                                            width)  # Shape: (batch_size, num_instances, num_channel, height, width)

        # Head2: KDE
        embeddings_fc = self.ucc_model.kde_embeddings(embeddings_conv)
        embeddings_reshaped = embeddings_fc.view(batch_size, num_instances, embeddings_fc.shape[
            -1])  # Shape: (batch_size, num_instances, embedding_size)
        feature_distribution = self.ucc_model.kde(embeddings_reshaped, self.ucc_model.num_bins,
                                                  self.ucc_model.sigma)  # Shape: (batch_size, num_bins * embedding_size)

        logits = self.ucc_model.mlp_classifier(feature_distribution)

        return logits, decoded_img

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits, decoded_img = self.forward_pass(x)
        ucc_loss, ae_loss, train_loss = self.ucc_model.loss_function_multihead(logits, decoded_img, y, x)

        self.log_dict({"train_loss": train_loss, "ucc_loss": ucc_loss, "ae_loss": ae_loss})

        return train_loss

    def validation_step(self, batch, batch_idx):
        """
        Each validation_step processes a single batch, and the results are automatically aggregated over all batches.
        """
        x, y = batch
        logits, decoded_img = self.forward_pass(x)

        ucc_loss, ae_loss, val_loss = self.ucc_model.loss_function_multihead(logits, decoded_img, y, x)

        # Compute accuracy
        # Convert logits to predicted classes
        preds = torch.argmax(logits, dim=1)
        correct_count = torch.sum(preds == y)
        val_acc = correct_count.float() / y.size(0)

        print(f"\n{y} (y)")
        print(f"\n{preds} (preds) ")

        # logging
        self.log_dict({
            "val_loss": val_loss,
            "val_acc": val_acc,
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


############################################
# RUN
############################################

if __name__ == "__main__":
    DATA_CONFIG = DataConfig(
        lower_ucc=1,
        upper_ucc=4,
        bag_size=20,
        bag_fraction=1,
        batch_size=16,
        transform={
            'random_horizontal_flip': lambda img: TF.hflip(img) if random.random() > 0.5 else img,
            # 'random_vertical_flip': lambda img: TF.vflip(img) if random.random() > 0.5 else img,
            # 'color_jitter': lambda img: TF.adjust_brightness(img, brightness_factor=random.uniform(0.8, 1.2)),
            'normalize': lambda img: TF.normalize(img, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        }
    )

    MODEL_CONFIG = ModelConfig(
        num_bins=10,
        sigma=0.1,
        dropout_rate=0.1,
        num_classes=5,
        embedding_size=110,
        fc2_size=512
    )

    # Data
    dataloaders = create_dataloaders(**DATA_CONFIG.__dict__)

    # Model + Optim
    model = UCCModel(**MODEL_CONFIG.__dict__)
    ucc_module = UCCModule(model, 0.0001, 0.5)

    # Trainer
    trainer = L.Trainer(
        max_steps=1000,
        default_root_dir="../logs"
    )

    # Run
    trainer.fit(
        model=ucc_module,
        train_dataloaders=dataloaders['train'],
        val_dataloaders=dataloaders['val']
    )
