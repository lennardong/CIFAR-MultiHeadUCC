import os
from math import nan

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm


def loss_function(logits, decoded_img, labels, original_imgs, ucc_loss_weight=0.5):

    ae_loss_weight = 1 - ucc_loss_weight

    ucc_loss = F.cross_entropy(logits, labels)
    ae_loss = F.mse_loss(decoded_img, original_imgs)
    combined_loss = (ucc_loss_weight * ucc_loss) + (ae_loss_weight * ae_loss)

    return combined_loss


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 model_dir: str,
                 model_name: str,
                 total_steps: int,
                 eval_interval: int,
                 ucc_loss_weight: float = 0.5):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name
        self.total_steps = total_steps
        self.eval_interval = eval_interval
        self.ucc_loss_weight = ucc_loss_weight
        self.best_eval_acc = 0

        # log
        self.train_loss_list = []
        self.val_loss_list = []
        self.val_acc_list = []
        self.step_list = []

        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def train(self):
        step = 0
        train_loss_accum = 0

        print("##########################\n"
              "# Starting Training...\n"
              "#########################")

        pbar = tqdm(total=self.total_steps, desc='Training Progress')

        while step < self.total_steps:
            for batch_samples, batch_labels in self.train_loader:
                # Move data to the device
                batch_samples = batch_samples.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                logits, decoded_img = self.model(batch_samples)
                loss = self.loss_function(logits, decoded_img, batch_labels, batch_samples, self.ucc_loss_weight)
                train_loss_accum += loss.item()

                # Backprop
                loss.backward()
                self.optimizer.step()

                if step % self.eval_interval == 0 and step > 0:
                    train_loss = train_loss_accum / self.eval_interval
                    val_loss, val_acc = self.evaluate()
                    train_loss_accum = 0  # Reset accumulator

                    # Log
                    self.train_loss_list.append(train_loss)
                    self.val_loss_list.append(val_loss)
                    self.val_acc_list.append(val_acc)
                    self.step_list.append(step)
                    self.save_model(val_acc)

                    self.best_eval_acc = val_acc if val_acc > self.best_eval_acc else self.best_eval_acc

                    if train_loss == nan or val_loss == nan or val_acc == nan:
                        print("NaN loss detected. Exiting training...")
                        break

                    # Update progress bar
                    pbar.set_postfix_str(
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    pbar.update(self.eval_interval)

                step += 1
                if step >= self.total_steps:
                    break

        pbar.close()
        print(f"Training completed. Best validation accuracy: {self.best_eval_acc:.4f}")

    def evaluate(self):
        self.model.eval()
        val_loss_list = []
        val_acc_list = []
        with torch.no_grad():
            for batch_samples, batch_labels in self.val_loader:
                batch_samples = batch_samples.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Get model outputs
                logits, decoded_imgs = self.model(batch_samples)

                # Compute metrics
                val_loss = self.loss_function(logits, decoded_imgs, batch_labels, batch_samples, self.ucc_loss_weight)

                _, preds = torch.max(logits, dim=1)
                acc = (preds == batch_labels).float().mean().item()

                val_acc_list.append(acc)
                val_loss_list.append(val_loss.item())

        self.model.train()  # Switch back to training mode
        return np.mean(val_loss_list), np.mean(val_acc_list)

    def save_model(self, eval_acc):
        # Save best model
        if eval_acc > self.best_eval_acc:
            self.best_eval_acc = eval_acc
            save_path = os.path.join(self.model_dir, f"{self.model_name}_best.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_path)

        # Save artefacts
        artefacts_path = os.path.join(self.model_dir, f"{self.model_name}_artefacts.json")
        with open(artefacts_path, "w") as f:
            json.dump(
                {
                    "train_loss": self.train_loss_list,
                    "val_loss": self.val_loss_list,
                    "val_acc": self.val_acc_list,
                    "step": self.step_list,
                    "best_eval_acc": self.best_eval_acc,
                }, f, indent=4
            )

    @staticmethod
    def loss_function(logits, decoded_img, labels, original_imgs, ucc_loss_weight=0.5):

        ae_loss_weight = 1 - ucc_loss_weight

        ucc_loss = F.cross_entropy(logits, labels)
        ae_loss = F.mse_loss(decoded_img, original_imgs)
        combined_loss = (ucc_loss_weight * ucc_loss) + (ae_loss_weight * ae_loss)

        return combined_loss

# Example usage:
# trainer = Trainer(model, optimizer, train_loader, val_loader, device, "path_to_model_dir", "model_name", 10000, 100)
# trainer.train()
