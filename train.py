# train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from utils.data_utils import get_data_loaders
from utils.model_utils import build_model
from utils.train_utils import train_epoch, eval_epoch, EarlyStopping
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


def main():
    parser = argparse.ArgumentParser(description="Train TCN on EMG dataset with CV splitting and metrics logging")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory of segmented dataset (folders per class)')
    parser.add_argument('--split', type=int, default=0,
                        help='Cross-validation split index (0 to 4)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for train/val/test loaders')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for optimizer')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of subprocesses for data loading')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Compute device')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience on validation accuracy')
    parser.add_argument('--wandb_project', type=str, default='EMG_TCN',
                        help='WandB project name for logging')
    args = parser.parse_args()

    # Initialize Weights & Biases
    wandb.login(key='dbefa42bdc21e918944847403b184f3084b373e6')
    wandb.init(project=args.wandb_project, config=vars(args))
    device = torch.device(int(args.device))

    # Data loaders for specified split
    train_loader, val_loader, test_loader, label_map = get_data_loaders(
        dataset_dir=args.data_dir,
        split_idx=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    num_classes = len(label_map)
    print(f"Gesture classes ({num_classes}): {label_map}")

    # Build model
    model = build_model(in_channels=64, num_classes=num_classes, device=device)
    wandb.watch(model, log='all', log_freq=10)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.tensor((1,1,1,0.1,1,1,1,1)).to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Early stopping
    early_stopper = EarlyStopping(patience=args.patience, mode='max', delta=0.0)

    best_model_path = os.path.join(args.data_dir, f"tcn_split{args.split}_best.pt")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Train, validate, test
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics     = eval_epoch(model, val_loader, criterion, device, split_name='val')
        test_loss, test_metrics   = eval_epoch(model, test_loader, criterion, device, split_name='test')

        print(f"Epoch {epoch:02d}: "
              f"Train acc={train_metrics['accuracy']:.4f}, "
              f"Val acc={val_metrics['accuracy']:.4f}, "
              f"Test acc={test_metrics['accuracy']:.4f}")

        # Check improvement and save best
        current_val_acc = val_metrics['accuracy']
        if early_stopper is not None:
            early_stopper(current_val_acc)
            if current_val_acc >= early_stopper.best_score:
                torch.save(model.state_dict(), best_model_path)
                wandb.save(best_model_path)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    final_test_loss, final_test_metrics = eval_epoch(model, test_loader, criterion, device, split_name='test')
    print(f"Final Test Acc: {final_test_metrics['accuracy']:.4f}")

    wandb.finish()


if __name__ == '__main__':
    main()
