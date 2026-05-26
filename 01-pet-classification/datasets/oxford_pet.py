import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2 as transforms

from configs import config

# =========================================================
# TRANSFORMS
# =========================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int):

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.8, 1.0),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
            ),
            transforms.RandomPerspective(
                distortion_scale=0.2,
                p=0.3,
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# =========================================================
# DATASETS
# =========================================================


def get_eval_transforms(image_size: int):

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_datasets(config: config.DatasetConfig):
    train_transform = get_train_transforms(config.image_size)
    eval_transform = get_eval_transforms(config.image_size)
    full_train_dataset = OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="category",
        download=True,
        transform=train_transform,
    )
    val_dataset_full = OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="category",
        download=False,
        transform=eval_transform,
    )
    test_dataset = OxfordIIITPet(
        root="./data",
        split="test",
        target_types="category",
        download=False,
        transform=eval_transform,
    )

    train_size = int(config.train_split * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    generator = torch.Generator().manual_seed(config.seed)

    train_indices, val_indices = random_split(
        range(len(full_train_dataset)),
        [train_size, val_size],
        generator=generator,
    )

    train_dataset = torch.utils.data.Subset(
        full_train_dataset,
        train_indices.indices,
    )

    val_dataset = torch.utils.data.Subset(
        val_dataset_full,
        val_indices.indices,
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(batch_size, image_size):
    train_dataset, val_dataset, test_dataset = get_datasets(image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=4,
    )
    return train_loader, val_loader, test_loader
