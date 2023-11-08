# Utils
from typing import Dict, Tuple, List
from itertools import combinations

# Maths
import numpy as np
import torch

# HF helpers
from datasets import Dataset, DatasetDict

# PyTorch

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

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


def create_ucc_datasets(data_splits: Dict[str, Dict[str, np.ndarray]],
                        ucc_bounds: Tuple[int, int]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Creates datasets with bags for CIFAR10 for each data split, with bags labeled by length of UCC,
    using simple dictionaries and NumPy arrays.

    Output:
    e.g. for UCC combination [1,2,4] -> 3: [tensor1, tensor2, ...., tensorN], where each array is an image with label 1, 2 or 4
    """
    lower_bound, upper_bound = ucc_bounds
    datasets_with_bags = {}

    # Generate UCC combinations for the specified bounds
    ucc_combinations = generate_ucc_combinations(lower_bound, upper_bound)

    for split, split_data in data_splits.items():
        images = split_data['images']
        labels = split_data['labels']

        # Create a list to hold all bags and their corresponding labels for the current split
        bags = []
        bag_labels = []

        # Create bags based on UCC combinations
        for ucc in ucc_combinations:
            # Find indices of all images belonging to classes in the current UCC combination
            indices = np.concatenate([np.flatnonzero(labels == class_label) for class_label in ucc])

            # Select all images for the current UCC combination
            ucc_images = images[indices]

            # Convert images to a PyTorch tensor, if you prefer tensors over numpy arrays
            ucc_images_tensor = torch.tensor(ucc_images)

            # Add images and the UCC label to the lists
            bags.append(ucc_images_tensor)
            bag_labels.append(len(ucc))

        # Convert the labels to a PyTorch tensor
        bag_labels_tensor = torch.tensor(bag_labels, dtype=torch.long)

        # Create a simple dictionary for the current split
        datasets_with_bags[split] = {
            'images': bags,  # or bags_np if you prefer to keep as numpy arrays
            'labels': bag_labels_tensor
        }

    return datasets_with_bags

def create_ucc_datasets_hf(data_splits: Dict[str, Dict[str, np.ndarray]],
                        ucc_bounds: Tuple[int, int]) -> Dict[str, Dataset]:
    """
    Creates datasets with bags for CIFAR10 for each data split, with bags labeled by length of UCC.
    """
    lower_bound, upper_bound = ucc_bounds
    datasets_with_bags = {}

    # Generate UCC combinations for the specified bounds
    ucc_combinations = generate_ucc_combinations(lower_bound, upper_bound)

    for split, split_data in data_splits.items():
        images = split_data['images']
        labels = split_data['labels']

        # Create a list to hold all bags and their corresponding labels for the current split
        bags = []
        bag_labels = []

        # Create bags based on UCC combinations
        for ucc in ucc_combinations:

            # Find indices of all images belonging to classes in the current UCC combination
            indices = []
            for class_label in ucc:
                # find indices of all images belonging to the current class label
                class_indices = np.where(labels == class_label)[0]
                indices.extend(class_indices)

            # Select all images for the current UCC combination
            indices = np.array(indices)
            ucc_images = images[indices]

            # Add images and the UCC label to the lists
            bags.append(ucc_images)
            bag_labels.append(len(ucc))

        # Convert the lists to numpy arrays (this step may not be necessary depending on the dataset format)
        bags_np = np.array(bags, dtype=object)  # dtype=object since bags can have different sizes
        bag_labels_np = np.array(bag_labels)

        # Create a Hugging Face Dataset for the current split
        datasets_with_bags[split] = Dataset.from_dict({
            'images': bags_np,
            'labels': bag_labels_np
        })
        datasets_with_bags[split].set_format(type='torch', columns=['images', 'labels'])

    return datasets_with_bags

# def create_hf_datasets(data_splits: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dataset]:
#     """
#     Converts NumPy data splits into Hugging Face datasets.
#
#     Parameters:
#         data_splits (Dict[str, Dict[str, np.ndarray]]): Dictionary containing the data splits with images and labels.
#
#     Returns:
#         Dict[str, Dataset]: A dictionary of Hugging Face datasets for each split.
#     """
#     datasets = {}
#     for split in data_splits:
#         datasets[split] = Dataset.from_dict({
#             'images': data_splits[split]['images'],
#             'labels': data_splits[split]['labels']
#         })
#         datasets[split].set_format(type='torch', columns=['images', 'labels'])
#     return datasets

# def create_bags(dataset: Dataset, bag_size_: int, num_labels_: int, num_bags_: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
#     """
#     Creates bags of data by sampling images from the dataset.
#
#     Parameters:
#         dataset (Dataset): The Hugging Face dataset to sample from.
#         bag_size_ (int): The number of images per bag.
#         num_labels_ (int): The number of unique labels per bag.
#         num_bags_ (int): The total number of bags to create.
#
#     Returns:
#         Tuple[List[np.ndarray], List[np.ndarray]]: Two lists containing the bags and their corresponding labels.
#     """
#     bags = []
#     labels = []
#     for _ in range(num_bags_):
#         chosen_labels = np.random.choice(range(10), num_labels_, replace=False)
#         bag = []
#         bag_labels = []
#         for label in chosen_labels:
#             label_instances = dataset.filter(lambda example: example['labels'] == label)
#             sampled_instances = label_instances.shuffle().select(range(bag_size_))
#             bag.extend(sampled_instances['images'])
#             bag_labels.extend(sampled_instances['labels'])
#         bags.append(bag)
#         labels.append(bag_labels)
#     return bags, labels




# def create_datasets_with_bags(data_splits: Dict[str, Dataset],
#                               lower_bound: int, upper_bound: int) -> Dict[str, Dataset]:
#     """
#     Creates datasets with bags for CIFAR10 for each data split, with bags labeled by UCC.
#     """
#     datasets_with_bags = {}
#
#     # Generate UCC combinations for the specified bounds
#     ucc_combinations = generate_ucc_combinations(lower_bound, upper_bound)
#
#     for split, tensor_dataset in data_splits.items():
#         images, labels = tensor_dataset.tensors
#
#         bags = []
#         bag_labels = []
#
#         for ucc in ucc_combinations:
#             # Select all images that belong to any class in the current UCC combination
#             selected_indices = torch.cat([torch.where(labels == class_idx)[0] for class_idx in ucc]).tolist()
#             selected_images = images[selected_indices]
#
#             bags.append(selected_images)
#             bag_labels.append(len(ucc))
#
#         # Convert the list of tensors to a single tensor for images
#         bags = torch.cat(bags, dim=0)
#
#         # Convert labels to a tensor
#         bag_labels = torch.tensor(bag_labels, dtype=torch.long)
#
#         # Create the Hugging Face Dataset for the current split
#         datasets_with_bags[split] = Dataset.from_dict({
#             'images': bags,
#             'labels': bag_labels
#         })
#         datasets_with_bags[split].set_format(type='torch', columns=['images', 'labels'])
#
#     return datasets_with_bags

# def create_datasets_with_bags_v1(data_splits: Dict[str, Dataset], bag_size_: int, num_labels_: int, num_bags_: int) -> Dict[str, Dataset]:
#     """
#     Creates datasets with bags for each data split.
#
#     Parameters:
#         data_splits (Dict[str, Dataset]): Datasets for train, val, and test splits.
#         bag_size_ (int): The number of images per bag.
#         num_labels_ (int): The number of unique labels per bag.
#         num_bags_ (int): The total number of bags to create.
#
#     Returns:
#         Dict[str, Dataset]: Datasets with bags for each split.
#     """
#     datasets_with_bags = {}
#     for split, dataset in data_splits.items():
#         bags, bag_labels = create_bags(dataset, bag_size_, num_labels_, num_bags_)
#         datasets_with_bags[split] = Dataset.from_dict({'bags': bags, 'labels': bag_labels})
#         datasets_with_bags[split].set_format(type='torch', columns=['bags', 'labels'])
#     return datasets_with_bags


def apply_transforms_to_bags(datasets_with_bags: Dict[str, Dataset], transforms: Compose) -> Dict[str, Dataset]:
    """
    Applies transformations to each dataset split.

    Parameters:
        datasets_with_bags (Dict[str, Dataset]): Datasets with bags to transform.
        transforms (Compose): The Compose object containing torchvision transforms.

    Returns:
        Dict[str, Dataset]: The datasets with the transformations applied.
    """
    for split in datasets_with_bags:
        datasets_with_bags[split] = datasets_with_bags[split].map(lambda example: {'bags': transforms(example['bags'])}, batched=True)
    return datasets_with_bags


def get_data_loaders(datasets_with_bags: Dict[str, Dataset], batch_size_: int = 32) -> Dict[str, DataLoader]:
    """
    Converts datasets into DataLoader objects.

    Parameters:
        datasets_with_bags (Dict[str, Dataset]): Datasets with bags to convert into DataLoader.
        batch_size_ (int): Batch size for the DataLoader.

    Returns:
        Dict[str, DataLoader]: DataLoaders for each dataset split.
    """
    _loaders = {}
    for split, dataset in datasets_with_bags.items():
        _loaders[split] = DataLoader(dataset, batch_size=batch_size_, shuffle=True if split == 'train' else False)
    return _loaders


############################################
# Main function to orchestrate the loading, bag creation, and transformation process
############################################


def main(npz_path_, bag_size_, num_labels_, num_bags_, batch_size_):

    # Define custom transforms (augmentation)
    transforms = Compose([
        RandomHorizontalFlip(),  # Example augmentation
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR-10 normalization
    ])

    data_splits = load_npz_data(npz_path_)
    hf_datasets = create_hf_datasets(data_splits)
    datasets_with_bags = create_datasets_with_bags(hf_datasets )
    transformed_datasets = apply_transforms_to_bags(datasets_with_bags, transforms)
    _loaders = get_data_loaders(transformed_datasets, batch_size_)

    return _loaders


############################################
# Execution
############################################


if __name__ == '__main__':
    # Call the main function with the desired parameters
    npz_path = '../data/Splitted CIFAR10.npz'
    bag_size = 5  # Number of images per bag
    num_labels = 3  # Number of unique labels per bag
    num_bags = 100  # Number of bags to create
    batch_size = 32  # Batch size for DataLoader

    loaders = main(npz_path, bag_size, num_labels, num_bags, batch_size)
