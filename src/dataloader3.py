# Standard library imports
from typing import Dict, List, Tuple
from itertools import combinations
import random

# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader

# Data handling import
import numpy as np

############################################
# HELPER FUNCTIONS
############################################

def load_npz_data(npz_path_: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads data from a .npz file and organizes it into a dictionary format.
    """
    with np.load(npz_path_) as npz_file:
        return {
            'train': {'images': npz_file['x_train'], 'labels': npz_file['y_train']},
            'val': {'images': npz_file['x_val'], 'labels': npz_file['y_val']},
            'test': {'images': npz_file['x_test'], 'labels': npz_file['y_test']}
        }


def generate_ucc_combinations(lower_bound: int, upper_bound: int) -> List[List[int]]:
    """
    Generates all combinations of class labels within a specified range.
    """
    return [list(comb) for i in range(lower_bound, upper_bound + 1) for comb in combinations(range(10), i)]


class CIFAR10BagsDataset(Dataset):
    """
    A custom PyTorch Dataset for creating bags of CIFAR-10 images based on UCC combinations.
    Each bag will contain a fixed total number of images, distributed across the classes in the combination.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, ucc_combinations: List[List[int]],
                 total_images_per_bag: int, bag_fraction: float):
        # Calculate the number of bags needed based on the fraction of the total combinations
        self.num_bags = int(len(ucc_combinations) * bag_fraction)
        self.images = images
        self.labels = labels
        self.total_images_per_bag = total_images_per_bag
        # Precompute the bags to speed up __getitem__
        self.bags, self.bag_labels = self.precompute_bags(ucc_combinations)

    def precompute_bags(self, ucc_combinations: List[List[int]]) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Precomputes the bags based on UCC combinations.
        """
        bags = []
        bag_labels = []
        # Randomly sample combinations to form the bags
        for _ in range(self.num_bags):
            combination = random.choice(ucc_combinations)
            images_per_class = self.total_images_per_bag // len(combination)
            remainder = self.total_images_per_bag % len(combination)

            bag_images = []
            for i, class_label in enumerate(combination):
                # Calculate the number of images to sample for this class
                num_images = images_per_class + (1 if i < remainder else 0)
                class_indices = np.where(self.labels == class_label)[0]
                if num_images > len(class_indices):
                    raise ValueError(f"Not enough images to sample for class {class_label}")
                sampled_indices = np.random.choice(class_indices, size=num_images, replace=False)
                bag_images.append(self.images[sampled_indices])

            # Combine images from all classes in the combination to form the bag
            bag = np.concatenate(bag_images, axis=0)
            bags.append(torch.tensor(bag, dtype=torch.float32))
            bag_labels.append(len(combination))
        return bags, bag_labels

    def __len__(self):
        return self.num_bags

    # def __getitem__(self, idx):
    #     # Return the precomputed bag and its label
    #     return self.bags[idx], torch.tensor(self.bag_labels[idx], dtype=torch.long)

    # def __getitem__(self, idx):
    #     # Retrieve the precomputed bag and its label
    #     bag, label = self.bags[idx], self.bag_labels[idx]
    #
    #     # Ensure the bag tensor is of shape [num_instances, num_channel, patch_size, patch_size]
    #     # If the images are stored as [patch_size, patch_size, num_channel], we need to transpose them
    #     bag = bag.view(-1, 32, 32, 3)  # Here we assume each image is 32x32x3
    #     bag = bag.permute(0, 3, 1, 2)  # Rearrange dimensions to [num_instances, num_channel, patch_size, patch_size]
    #
    #     return bag, torch.tensor(label, dtype=torch.long)

    def __getitem__(self, idx):
        # Retrieve the precomputed bag and its label
        bag, label = self.bags[idx], self.bag_labels[idx]

        # Infer the shape from the bag itself instead of hardcoding
        # Assuming bag is a flat array of image data
        num_channels = self.images.shape[-1]  # Typically 3 for RGB, 1 for grayscale
        height, width = self.images.shape[1:3]  # Infer the height and width

        # Calculate the number of instances by dividing the total number of pixels by the pixels per image
        num_instances = bag.numel() // (num_channels * height * width)

        # Reshape bag to [num_instances, height, width, num_channels]
        bag = bag.view(num_instances, height, width, num_channels)

        # Permute the dimensions to [num_instances, num_channels, height, width] to match PyTorch's expectation
        bag = bag.permute(0, 3, 1, 2)

        return bag, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """
    Custom collate function for combining bags into a batch.
    """
    bags, labels = zip(*batch)
    # Stack the bags along a new dimension to create a batch
    bags = torch.stack(bags)
    labels = torch.tensor(labels, dtype=torch.long)
    return bags, labels

def create_dataloaders(npz_path_: str, ucc_combinations: List[List[int]], bag_size: int, bag_fraction: float) -> Dict[str, DataLoader]:
    """
    Creates PyTorch DataLoaders for CIFAR-10 data splits based on UCC combinations.
    """
    data_splits = load_npz_data(npz_path_)
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        dataset = CIFAR10BagsDataset(
            data_splits[split]['images'],
            data_splits[split]['labels'],
            ucc_combinations,
            bag_size,
            bag_fraction
        )
        # Specify the custom collate function for the DataLoader
        dataloaders[split] = DataLoader(dataset, batch_size=12, shuffle=True, collate_fn=collate_fn)

    return dataloaders


# Example usage
if __name__ == "__main__":
    npz_path = '../data/Splitted CIFAR10.npz'
    lower_bound = 2
    upper_bound = 4
    bag_size = 500  # The number of images per bag
    bag_fraction = 0.1  # Fraction of UCC combinations to use

    ucc_combinations = generate_ucc_combinations(lower_bound, upper_bound)
    dataloaders = create_dataloaders(npz_path, ucc_combinations, bag_size, bag_fraction)

    # Testing the dataloaders
    for images, labels in dataloaders['train']:
        print(f'Images batch shape: {images.shape}')
        print(f'Labels batch shape: {labels.shape}')
        print(f'Labels: {labels}')
        break

# TODO : fix upper and lower bound to sample fully from 0~9
# TODO: implement image augmentations