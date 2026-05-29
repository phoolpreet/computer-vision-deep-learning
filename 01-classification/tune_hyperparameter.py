import optuna
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from engine.train import train_epoch
from engine.evaluate import evaluate

# =========================================================
# Generic Trainer Config
# =========================================================

NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Generic Objective Function
# =========================================================


def objective(
    trial,
    model_fn,
    dataloader_fn,
    optimizer_fn=None,
    scheduler_fn=None,
    criterion_fn=None,
):
    """
    Generic Optuna objective.

    Parameters
    ----------
    trial : optuna.trial.Trial

    model_fn : callable
        Function that returns model.
        Example:
            lambda trial: ResNet(dropout=...)

    dataloader_fn : callable
        Function that returns:
            train_loader, val_loader

    optimizer_fn : callable
        Function:
            optimizer_fn(trial, model)

    scheduler_fn : callable
        Function:
            scheduler_fn(trial, optimizer, train_loader)

    criterion_fn : callable
        Returns loss function
    """

    # =====================================================
    # Dataloaders
    # =====================================================

    train_loader, val_loader = dataloader_fn(trial)

    # =====================================================
    # Model
    # =====================================================

    model = model_fn(trial).to(DEVICE)

    # =====================================================
    # Criterion
    # =====================================================

    if criterion_fn is None:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = criterion_fn()

    # =====================================================
    # Optimizer
    # =====================================================

    if optimizer_fn is None:

        lr = trial.suggest_float(
            "learning_rate",
            1e-5,
            1e-3,
            log=True,
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
        )

    else:
        optimizer = optimizer_fn(trial, model)

    # =====================================================
    # Scheduler
    # =====================================================

    scheduler = None

    if scheduler_fn is not None:
        scheduler = scheduler_fn(
            trial,
            optimizer,
            train_loader,
        )

    # =====================================================
    # Training Loop
    # =====================================================

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):

        train_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
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
            model=model,
            dataloader=val_loader,
            loss_fn=criterion,
            device=DEVICE,
        )

        # =================================================
        # Report to Optuna
        # =================================================

        trial.report(val_acc, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        best_val_acc = max(best_val_acc, val_acc)

        print(f"Trial={trial.number} | " f"Epoch={epoch+1} | " f"Val Acc={val_acc:.4f}")

    return best_val_acc


# =========================================================
# Example Usage
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Dataset Function
    # -----------------------------------------------------

    from datasets.oxford_pet import get_dataloaders

    def dataloader_fn(trial):

        batch_size = trial.suggest_categorical(
            "batch_size",
            [16, 32, 64],
        )

        train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

        return train_loader, val_loader

    # -----------------------------------------------------
    # Model Function
    # -----------------------------------------------------

    from models.classifier_resnet import ClassifierResNet
    from models.classifier_mobilenet import ClassifierMobileNet

    def model_fn(trial):

        dropout = trial.suggest_float(
            "dropout",
            0.1,
            0.5,
        )

        return ClassifierMobileNet(num_classes=37, dropout=dropout)

    # -----------------------------------------------------
    # Optimizer Function
    # -----------------------------------------------------

    def optimizer_fn(trial, model):

        lr = trial.suggest_float(
            "learning_rate",
            1e-5,
            1e-3,
            log=True,
        )

        optimizer_name = trial.suggest_categorical(
            "optimizer",
            ["Adam", "AdamW", "SGD"],
        )

        if optimizer_name == "Adam":
            return optim.Adam(
                model.parameters(),
                lr=lr,
            )

        elif optimizer_name == "AdamW":
            return optim.AdamW(
                model.parameters(),
                lr=lr,
            )

        elif optimizer_name == "SGD":

            momentum = trial.suggest_float(
                "momentum",
                0.8,
                0.99,
            )

            return optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
            )

    # -----------------------------------------------------
    # Scheduler Function
    # -----------------------------------------------------

    def scheduler_fn(
        trial,
        optimizer,
        train_loader,
    ):

        max_lr = trial.suggest_float(
            "max_lr",
            1e-3,
            1e-1,
            log=True,
        )

        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=len(train_loader) * NUM_EPOCHS,
        )

    # =====================================================
    # Study
    # =====================================================

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(
        lambda trial: objective(
            trial=trial,
            model_fn=model_fn,
            dataloader_fn=dataloader_fn,
            optimizer_fn=optimizer_fn,
            scheduler_fn=scheduler_fn,
        ),
        n_trials=20,
    )

    # =====================================================
    # Results
    # =====================================================

    print("\n========== BEST TRIAL ==========")

    print(f"Best Accuracy: {study.best_value:.4f}")

    print("\nBest Hyperparameters:")

    for key, value in study.best_params.items():
        print(f"{key}: {value}")
