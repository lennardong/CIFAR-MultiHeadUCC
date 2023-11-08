# Standard library imports
from typing import Dict, List, Tuple
from itertools import combinations
import random

# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F

# Data handling import
import numpy as np

############################################
# HELPER FUNCTIONS
############################################


def load_npz_data(npz_path_: str) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Loads data from a .npz file, normalizes the images, and organizes them into a dictionary format.

    Parameters:
        npz_path_ (str): The file path to the .npz file containing CIFAR-10 data.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: A dictionary with train, val, and test splits, each containing normalized images and labels.
    """
    with np.load(npz_path_) as npz_file:
        # Convert images to torch.Tensor and normalize them to the range [0, 1]
        x_train = torch.tensor(npz_file['x_train'], dtype=torch.float32) / 255.0
        x_val = torch.tensor(npz_file['x_val'], dtype=torch.float32) / 255.0
        x_test = torch.tensor(npz_file['x_test'], dtype=torch.float32) / 255.0

        return {
            'train': {'images': x_train, 'labels': torch.tensor(npz_file['y_train'])},
            'val': {'images': x_val, 'labels': torch.tensor(npz_file['y_val'])},
            'test': {'images': x_test, 'labels': torch.tensor(npz_file['y_test'])}
        }


def generate_ucc_combinations(lower_bound_: int, upper_bound_: int) -> List[List[int]]:
    """
    Generates all combinations of class labels within a specified range.
    """
    return [list(comb) for i in range(lower_bound_, upper_bound_ + 1) for comb in combinations(range(10), i)]


def collate_fn(batch):
    """
    Custom collate function for combining bags into a batch.
    """
    bags, labels = zip(*batch)
    # Stack the bags along a new dimension to create a batch
    bags = torch.stack(bags)
    labels = torch.tensor(labels, dtype=torch.long)
    return bags, labels


############################################
# CLASS
############################################


class CIFAR10BagsDataset(Dataset):
    """
    A custom PyTorch Dataset for creating bags of CIFAR-10 images based on UCC combinations.
    Each bag will contain a fixed total number of images, distributed across the classes in the combination.
    """

    def __init__(self,
                 images: np.ndarray,
                 labels: np.ndarray,
                 ucc_combinations: List[List[int]],
                 total_images_per_bag: int,
                 bag_fraction: float,
                 transform: Dict = None):

        self.num_bags = int(len(ucc_combinations) * bag_fraction) # Number of bags to create
        self.images = images
        self.labels = labels
        self.total_images_per_bag = total_images_per_bag
        self.bags, self.bag_labels = self.precompute_bags(ucc_combinations) # Precompute bags to speed up __getitem__
        self.transform = transform

    def precompute_bags(self, ucc_combinations_: List[List[int]]) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Precomputes the bags based on UCC combinations.
        """
        bags = []
        bag_labels = []
        # Randomly sample combinations to form the bags
        for _ in range(self.num_bags):
            combination = random.choice(ucc_combinations_)
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

    # @staticmethod
    # def normalize_image(image):
    #     return (image.float() / 255.0 - torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)) / torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)

    def __len__(self):
        return self.num_bags

    def __getitem__(self, idx):
        # Retrieve the precomputed bag and its label
        bag, label = self.bags[idx], self.bag_labels[idx]

        # Apply transformations to each image in the bag
        if self.transform:
            transformed_bag = []
            for image in bag:
                # Permute the image tensor to have the channel dimension first
                image = image.permute(2, 0, 1)
                # image = self.normalize_image(image)

                # print(f'IMAGE IN BAG: {image.shape}') # DEBUG
                for _, operation in self.transform.items():
                    image = operation(image)
                transformed_bag.append(image)

            # Stack the list of transformed images into a 4D tensor
            bag = torch.stack(transformed_bag)

        return bag, torch.tensor(label, dtype=torch.long)


############################################
# MAIN
############################################

def create_dataloaders(
        npz_path_: str,
        lower_ucc: int,
        upper_ucc: int,
        bag_size: int,
        bag_fraction: float,
        transform: Dict = None) -> Dict[str, DataLoader]:
    """
    Creates PyTorch DataLoaders for CIFAR-10 data splits based on UCC combinations.
    """
    data_splits = load_npz_data(npz_path_)
    dataloaders = {}
    ucc_combinations = generate_ucc_combinations(lower_ucc, upper_ucc)

    for split in ['train', 'val', 'test']:
        dataset = CIFAR10BagsDataset(
            images=data_splits[split]['images'],
            labels=data_splits[split]['labels'],
            ucc_combinations=ucc_combinations,
            total_images_per_bag=bag_size,
            bag_fraction=bag_fraction,
            transform=transform
        )

        # Specify the custom collate function for the DataLoader
        dataloaders[split] = DataLoader(dataset, batch_size=12, shuffle=True, collate_fn=collate_fn)

    return dataloaders


if __name__ == "__main__":

    PATH = '../data/Splitted CIFAR10.npz'

    # Define a dictionary of transformations to apply.
    # use F so that it is direct transforms on tensor
    transforms = {
        'random_horizontal_flip': lambda img: F.hflip(img) if random.random() > 0.5 else img,
        'random_vertical_flip': lambda img: F.vflip(img) if random.random() > 0.5 else img,
        'color_jitter': lambda img: F.adjust_brightness(img, brightness_factor=random.uniform(0.8, 1.2)),
        'normalize': lambda img: F.normalize(img, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    }

    dataloaders = create_dataloaders(PATH, 2, 4, 300, 0.1, transforms)

    # Testing the dataloaders
    for images, labels in dataloaders['train']:
        print(f'Images batch shape: {images.shape}')
        print(f'Labels batch shape: {labels.shape}')
        print(f'Labels: {labels}')
        print(f'Images: {images}')
        break
