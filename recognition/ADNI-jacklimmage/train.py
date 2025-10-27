import torch
from torch import nn, optim
from dataset import load_data
from modules import *
import time, datetime

BATCH_SIZE = 2048
NUM_WORKERS = 4

NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
SAVE_PATH = "best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # --- Data Loading ---
    train_loader, test_loader = load_data(BATCH_SIZE, NUM_WORKERS)
    print(f"\nData loaders ready. Train samples: {len(train_loader.dataset)}, "
          f"Test samples: {len(test_loader.dataset)}")

    # --- Model Initialization ---
    model = GFNet(
        img_size=224, 
        patch_size=16, embed_dim=192, depth=6, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=2
    ).to(DEVICE)
    print(f"Model initialized on {DEVICE}.")

    # --- Loss Function, Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

    # --- Training ---
    train_losses = []
    val_losses = []
    val_accuracies = []
    max_accuracy = 0.0
    start_time = time.time()
    
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        val_acc_pct = val_acc * 100
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc_pct:.2f}%")

        # --- Save Best Model ---
        if val_acc_pct > max_accuracy:
            max_accuracy = val_acc_pct
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  *** Saved best model to {SAVE_PATH} with Val Acc: {max_accuracy:.2f}% ***")

    # --- The End ---
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('\nTraining finished.')
    print(f'Total training time: {total_time_str}')

if __name__ == "__main__":
    main()