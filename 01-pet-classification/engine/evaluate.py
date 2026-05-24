import torch
from sklearn import metrics


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = metrics.accuracy_score(all_labels, all_preds)
    epoch_precision = metrics.precision_score(all_labels, all_preds, average="weighted")
    epoch_recall = metrics.recall_score(all_labels, all_preds, average="weighted")
    cm = metrics.confusion_matrix(all_labels, all_preds)

    return (
        epoch_loss,
        epoch_acc,
        epoch_precision,
        epoch_recall,
        cm,
        all_preds,
        all_labels,
    )
