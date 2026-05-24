import os

import torch
import torch.optim.lr_scheduler as lr_scheduler
from configs import config
from datasets.oxford_pet import get_dataloaders
from engine.evaluate import evaluate
from engine.train import train_epoch
from models.pet_classifier_resnet import PetClassifierResNet
from utils import checkpoint, visualization

num_epochs = config.NUM_EPOCHS
initial_learning_rate = config.LEARNING_RATE
output_path = config.OUTPUT_PATH
max_lr = config.MAX_LR
dropout = config.DROPOUT
batch_size = config.BATCH_SIZE


def training(model, train_loader, val_loader):
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=len(train_loader) * num_epochs
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train one epoch
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
        )

        # Evaluate on validation set
        val_loss, val_acc, val_precision, val_recall, val_cm, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, ",
            f"Val Precision: {val_precision:.4f}, ",
            f"Val Recall: {val_recall:.4f} ",
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save locally
            checkpoint.save_checkpoint(
                model, os.path.join(output_path, "best_model.pth")
            )

    return train_losses, val_losses


if __name__ == "__main__":
    model = PetClassifierResNet(dropout=dropout)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size)
    train_losses, val_losses = training(model, train_loader, val_loader)
    visualization.plot_losses(train_losses, val_losses)
