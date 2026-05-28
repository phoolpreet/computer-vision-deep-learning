import os

import torch
from engine.evaluate import evaluate
from torch.utils.tensorboard import SummaryWriter

from utils import checkpoint


def train_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    scheduler,
    device,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        scheduler.step()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc



def train(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler,
    num_epochs,
    output_path,
    log_dir="log",
):

    writer = SummaryWriter(log_dir=os.path.join(output_path, log_dir))

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train one epoch
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scheduler,
            device,
        )

        # Evaluate on validation set
        val_loss, val_acc, val_precision, val_recall, val_cm, _, _ = evaluate(
            model,
            val_loader,
            loss_fn,
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

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        writer.add_scalar("Precision/Validation", val_precision, epoch)
        writer.add_scalar("Recall/Validation", val_recall, epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LearningRate", current_lr, epoch)

    writer.close()

    return train_losses, val_losses