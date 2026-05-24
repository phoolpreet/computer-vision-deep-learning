import optuna
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from datasets.oxford_pet import get_dataloaders
from engine.evaluate import evaluate
from engine.train import train_epoch
from models.pet_classifier_resnet import PetClassifierResNet

NUM_EPOCHS = 10  # keep small during tuning


def objective(trial):
    # =========================
    # Hyperparameters to search
    # =========================

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        1e-3,
        log=True,
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [16, 32, 64],
    )
    max_lr = trial.suggest_float(
        "max_lr",
        1e-3,
        1e-1,
        log=True,
    )

    dropout = trial.suggest_float(
        "dropout",
        0.1,
        0.5,
    )

    # =========================
    # Data
    # =========================

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # =========================
    # Device
    # =========================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # Model
    # =========================

    model = PetClassifierResNet(dropout=dropout).to(device)

    # =========================
    # Loss / Optimizer
    # =========================

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=len(train_loader) * NUM_EPOCHS,
    )

    # =========================
    # Training Loop
    # =========================

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):

        train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
        )

        (
            val_loss,
            val_acc,
            val_precision,
            val_recall,
            val_cm,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        # =========================
        # Report intermediate result
        # =========================

        trial.report(val_acc, epoch)

        # =========================
        # Early stopping / pruning
        # =========================

        if trial.should_prune():
            raise optuna.TrialPruned()

        best_val_acc = max(best_val_acc, val_acc)

        print(
            f"Trial={trial.number} | ",
            f"Epoch={epoch+1} | ",
            f"Val Acc={val_acc:.4f}",
        )

    return best_val_acc


if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(
        objective,
        n_trials=20,
    )
    print("\n========== BEST TRIAL ==========")

    print(f"Best Accuracy: {study.best_value:.4f}")

    print("\nBest Hyperparameters:")

    for key, value in study.best_params.items():
        print(f"{key}: {value}")
