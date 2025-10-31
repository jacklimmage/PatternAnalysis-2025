import torch
from torch import nn
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from modules import GFNet, validate_epoch
from dataset import load_data
from train import DEVICE, BATCH_SIZE, NUM_WORKERS, SAVE_PATH


def load_model():
    print("Initializing model structure...")
    # Model Initialization ---
    model = GFNet(
        img_size=224,
        patch_size=16, embed_dim=394, depth=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=2, drop_path_rate=0.3
    ).to(DEVICE)

    # --- Load Weights ---
    weights = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(weights)

    # Set model to evaluation mode (important for BatchNorm, Dropout, etc.)
    model.eval()
    print(f"Successfully loaded model from {SAVE_PATH} and set to evaluation mode.")
    return model


def compute_confusion_matrix(model, loader, device):
    """
    Compute confusion matrix (rows: true labels, cols: predicted labels) for the provided loader.
    Returns: (cm, class_names)
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1).cpu().numpy()
            preds.append(predicted)
            targets.append(labels.numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    num_classes = max(int(targets.max()), int(preds.max())) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1

    class_names = loader.dataset.classes

    return cm, class_names


def plot_confusion_matrix(cm, save_path, class_names):

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Tick marks and labels
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label', title="Confusion matrix")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with counts and percentages (percentage of total)
    total = cm.sum() if cm.sum() > 0 else 1
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            pct = (count / total) * 100.0
            label = f"{count}\n{pct:.1f}%"
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved confusion matrix image to {save_path}")
    else:
        plt.show()

def main():
    _, test_loader = load_data(BATCH_SIZE, NUM_WORKERS)

    model = load_model()

    criterion = nn.CrossEntropyLoss()
    loss, acc = validate_epoch(model, test_loader, criterion, DEVICE)
    print(f"\nModel Evaluation on Test Set - Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")

    # compute and plot confusion matrix (also get class names from the dataset)
    cm, class_names = compute_confusion_matrix(model, test_loader, DEVICE)
    plot_confusion_matrix(cm, "confusion_matrix.png", class_names)


if __name__ == "__main__":
    main()