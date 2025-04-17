# util/train_utils.py
import torch
import numpy as np
import wandb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score,
    recall_score
)


class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """
    def __init__(self, patience=5, mode='max', delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        else:
            improvement = score - self.best_score if self.mode == 'max' else self.best_score - score
            if improvement > self.delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True


def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute classification metrics given true labels, predicted labels, and prediction probabilities.

    Returns a dict with keys:
        'accuracy', 'f1', 'kappa', 'auc', 'sensitivity', 'specificity', 'confusion_matrix'
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = None
    cm = confusion_matrix(y_true, y_pred)
    sens = recall_score(y_true, y_pred, average='macro')
    # specificity per class = TN / (TN+FP)
    spec_per = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        spec_per.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    spec = float(np.mean(spec_per))

    return {
        'accuracy': float(acc),
        'f1': float(f1),
        'kappa': float(kappa),
        'auc': float(auc) if auc is not None else None,
        'sensitivity': float(sens),
        'specificity': spec,
        'confusion_matrix': cm.tolist(),
    }


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Runs one training epoch, logs metrics to WandB under 'train/'.
    Returns average loss and metrics dict.
    """
    model.train()
    all_true, all_pred, all_prob = [], [], []
    running_loss = 0.0
    n_samples = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_prob.append(probs)
        all_true.extend(labels.cpu().numpy())
        all_pred.extend(preds)

    avg_loss = running_loss / n_samples
    all_prob = np.vstack(all_prob)
    metrics = compute_metrics(np.array(all_true), np.array(all_pred), all_prob)
    metrics['loss'] = float(avg_loss)

    # log to wandb
    log_dict = {f"train/{k}": v for k, v in metrics.items() if v is not None}
    wandb.log(log_dict)

    return avg_loss, metrics


def eval_epoch(model, loader, criterion, device, split_name='val'):
    """
    Runs evaluation, logs metrics to WandB under '{split_name}/'.
    Returns average loss and metrics dict.
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    running_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            bs = labels.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_prob.append(probs)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds)

    avg_loss = running_loss / n_samples
    all_prob = np.vstack(all_prob)
    metrics = compute_metrics(np.array(all_true), np.array(all_pred), all_prob)
    metrics['loss'] = float(avg_loss)

    # log to wandb
    log_dict = {f"{split_name}/{k}": v for k, v in metrics.items() if v is not None}
    wandb.log(log_dict)

    return avg_loss, metrics
