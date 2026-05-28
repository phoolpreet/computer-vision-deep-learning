
import torch
import torch.optim.lr_scheduler as lr_scheduler

from configs import config
from datasets.oxford_pet import get_dataloaders
from engine.train import train
from engine.train import evaluate
from models.classifier_resnet import ClassifierResNet
from models.classifier_mobilenet import ClassifierMobileNet

if __name__ == "__main__":
    renet_config = config.get_config("resnet50")
    num_epochs = renet_config.num_epochs
    initial_learning_rate = renet_config.learning_rate
    output_path = renet_config.output_path
    max_lr = renet_config.max_lr
    dropout = renet_config.dropout

    # model = ClassifierResNet(dropout=dropout)
    model = ClassifierMobileNet(num_classes=37, dropout=dropout)


    train_loader, val_loader, test_loader = get_dataloaders()
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_learning_rate,
        weight_decay=1e-4,
    )
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=len(train_loader) * num_epochs,
    )

    train_losses, val_losses = train(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        scheduler,
        num_epochs,
        output_path,
        log_dir="log",
    )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # evaluate(model, test_loader, loss_fn, device)
