# Utils
from typing import Dict, Tuple, List
from itertools import combinations

# Maths
import numpy as np
import torch

# HF helpers
from datasets import Dataset, DatasetDict

# PyTorch
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from datasets import Dataset
import numpy as np
from typing import List, Dict, Tuple
from itertools import combinations
import random


# Assume load_npz_data and generate_ucc_combinations are available from dataloader.py

############################################
# HELPER FUNCTIONS
############################################

def load_npz_data(npz_path_: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads data from a .npz file and organizes it into a dictionary format.

    Parameters:
        npz_path_ (str): The file path to the .npz file containing CIFAR-10 data.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: A dictionary with train, val, and test splits, each containing images and labels.
    """
    npz_file = np.load(npz_path_)
    return {
        'train': {
            'images': npz_file['x_train'],
            'labels': npz_file['y_train']
        },
        'val': {
            'images': npz_file['x_val'],
            'labels': npz_file['y_val']
        },
        'test': {
            'images': npz_file['x_test'],
            'labels': npz_file['y_test']
        }
    }


def generate_ucc_combinations(lower_bound: int, upper_bound: int) -> List[List[int]]:
    """Generate all combinations of digits between lower_bound and upper_bound within class of 10
    each list item to later become a UCC bag, where UCC is length of liist

    Parameters:
            lower_bound: int. the lower bound of the range of digits to generate combinations from. inclusive
            upper_bound: int. the upper bound of the range of digits to generate combinations from. inclusive

    Returns:
        list of all combinations of digits between lower_bound and upper_bound within class of 10
    """
    ucc_combinations = []
    for i in range(lower_bound, upper_bound+1):
        ucc_combinations.extend(list(map(list, combinations(range(10), i))))

    return ucc_combinations


def create_balanced_bags(
        images: np.ndarray,
        labels: np.ndarray,
        ucc_combinations: List[List[int]],
        bag_size: int,
        bag_fraction: float) -> Tuple[List[np.ndarray], List[int]]:
    """
    Creates a list of balanced image bags and corresponding labels, sampling a fraction of UCC combinations.

    Args:
        images (np.ndarray): Array of images.
        labels (np.ndarray): Array of labels.
        ucc_combinations (List[List[int]]): UCC combinations.
        bag_size (int): The number of images per bag.
        bag_fraction (float): Fraction of UCC combinations to use (between 0 and 1).

    Returns:
        A tuple of bags and their labels.
    """
    # Determine the actual number of bags based on the fraction
    num_bags = int(len(ucc_combinations) * bag_fraction)
    selected_combinations = random.sample(ucc_combinations, num_bags)

    bags, bag_labels = [], []
    for combination in selected_combinations:
        bag_images = []
        for class_label in combination:
            class_indices = np.where(labels == class_label)[0]
            sampled_indices = np.random.choice(class_indices, size=bag_size, replace=False)
            bag_images.append(images[sampled_indices])
        bags.append(np.concatenate(bag_images, axis=0))
        bag_labels.append(len(combination))

    return bags, bag_labels

def create_datasets_with_bags(
        npz_path_: str,
        lower_bound: int,
        upper_bound: int,
        bag_size: int,
        bag_fraction: float) -> Dict[str, Dataset]:
    """
    Creates Hugging Face datasets with image bags for CIFAR-10 data splits based on a fraction of UCC combinations.

    Args:
        npz_path_ (str): Path to the NPZ file with CIFAR-10 data.
        lower_bound (int): Inclusive lower bound for UCC.
        upper_bound (int): Inclusive upper bound for UCC.
        bag_size (int): Number of images per bag.
        bag_fraction (float): Fraction of UCC combinations to use for creating bags.

    Returns:
        A dictionary of datasets for train, validation, and test splits.
    """
    data_splits = load_npz_data(npz_path_)
    ucc_combinations = generate_ucc_combinations(lower_bound, upper_bound)
    datasets = {}

    for split in ['train', 'val', 'test']:
        bags, bag_labels = create_balanced_bags(
            data_splits[split]['images'], data_splits[split]['labels'],
            ucc_combinations, bag_size, bag_fraction
        )
        flattened_images = np.concatenate(bags).astype(np.uint8)
        flattened_labels = np.repeat(bag_labels, bag_size).astype(np.uint8)
        dataset = Dataset.from_dict({'images': flattened_images, 'labels': flattened_labels})
        dataset.set_format(type='torch', columns=['images', 'labels'])
        datasets[split] = dataset

    return datasets

def apply_transforms(dataset):
    pass

############################################
# TESTS
############################################
"""
import torch
from typing import Iterable, Tuple

def check_data_shape(dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]], expected_shape: Tuple[int]):
    for images, _ in dataset:
        assert images.dim() == len(expected_shape), f"Data must be {len(expected_shape)}D"
        assert images.shape[1:] == expected_shape, f"Data shape must be {expected_shape}, but got {images.shape[1:]}"
        break  # Just check the first item for this validation

def check_data_type(dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]], expected_dtype: torch.dtype):
    for images, _ in dataset:
        assert images.dtype == expected_dtype, f"Data must be of type {expected_dtype}, but got {images.dtype}"
        break  # Just check the first item for this validation

def test_forward_pass(dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]], model):
    images, _ = next(iter(dataset))
    try:
        output = model(images)
        print("Forward pass successful. Output shape:", output.shape)
    except Exception as e:
        print("Forward pass failed:", e)

def check_label_distribution(dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]], lower_bound: int, upper_bound: int):
    for _, labels in dataset:
        assert all(lower_bound <= label <= upper_bound for label in labels), "Some labels are outside the specified UCC range"
        break  # Just check the first item for this validation

# Example usage in an IPython/Notebook:

# Assuming 'datasets' is a dictionary containing your train, val, and test datasets
# and 'YourModel' is your model class
batch_size, num_instances, num_channel, patch_size, patch_size = 4, 10, 3, 32, 32
# your_model = YourModel()

for split, dataset in datasets.items():
    print(f"Checking dataset: {split}")
    # Define the expected shape excluding the batch size
    expected_shape = (num_instances, num_channel, patch_size, patch_size)
    check_data_shape(dataset, expected_shape)
    check_data_type(dataset, torch.float32)
    check_label_distribution(dataset, lower_bound, upper_bound)
    test_forward_pass(dataset, your_model)
    print(f"Dataset {split} passed all checks.\n")
"""
############################################
# MAIN
############################################

# Example usage
# In your dataloader.py after creating the datasets
if __name__ == "__main__":
    npz_path = '../data/Splitted CIFAR10.npz'
    datasets = create_datasets_with_bags(npz_path, 2, 4, 32, 0.9)


"""
    # Perform checks
    for split, dataset in datasets.items():
        print(f"Checking dataset: {split}")
        check_data_shape(dataset, (num_instances, num_channel, patch_size, patch_size))
        check_data_type(dataset, torch.float32)
        check_label_distribution(dataset, lower_bound, upper_bound)
        # test_forward_pass(dataset, your_model)
        print(f"Dataset {split} passed all checks.\n")
"""