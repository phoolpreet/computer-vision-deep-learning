import torch
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2 as transforms


def get_transforms():
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transforms, val_test_transforms


def get_datasets():
    train_transforms, val_test_transforms = get_transforms()
    train_dataset = OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="category",
        download=True,
        transform=train_transforms,
    )
    test_dataset = OxfordIIITPet(
        root="./data",
        split="test",
        target_types="category",
        download=True,
        transform=val_test_transforms,
    )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    val_dataset.dataset.transform = val_test_transforms
    return train_dataset, val_dataset, test_dataset


def get_dataloaders():
    train_dataset, val_dataset, test_dataset = get_datasets()
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        drop_last=True,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        drop_last=True,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        drop_last=True,
        shuffle=False,
        num_workers=4,
    )
    return train_loader, val_loader, test_loader
