import os

import torch
from configs import config
from datasets.oxford_pet import get_dataloaders
from engine import evaluate
from models.pet_classifier_resnet import PetClassifierResNet
from utils import checkpoint


def evaluation(model, test_loader):
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    (
        test_loss,
        test_acc,
        test_precision,
        test_recall,
        test_cm,
        test_preds,
        test_labels,
    ) = evaluate.evaluate(model, test_loader, criterion, device)
    print("\n===== Test Results =====")

    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Test Precision : {test_precision:.4f}")
    print(f"Test Recall    : {test_recall:.4f}")


if __name__ == "__main__":
    pet_classifier = PetClassifierResNet()
    output_path = config.OUTPUT_PATH
    batch_size = config.BATCH_SIZE
    checkpoint.load_checkpoint(
        pet_classifier, os.path.join(output_path, "best_model.pth")
    )
    train_loader, val_loader, test_loader = get_dataloaders(batch_size)
    evaluation(pet_classifier, test_loader)
