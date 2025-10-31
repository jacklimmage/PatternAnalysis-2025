import torch
from torch import nn, optim
from dataset import load_data
from modules import *
import time, datetime

BATCH_SIZE = 256
NUM_WORKERS = 4

NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.2

SAVE_PATH = "best_model.pth"
PRETRAINED_PATH = "gfnet-xs-pretrained.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # --- Set Seeds ---
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # --- Data Loading ---
    train_loader, test_loader = load_data(BATCH_SIZE, NUM_WORKERS)
    print(f"\nData loaders ready. Train samples: {len(train_loader.dataset)}, "
          f"Test samples: {len(test_loader.dataset)}")

    # --- Model Initialization ---
    model = GFNet(
        img_size=224, 
        patch_size=16, embed_dim=394, depth=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=2, drop_path_rate=0.3
    ).to(DEVICE)
    print(f"Model initialized on {DEVICE}.")

    # --- Load Pretrained Weights ---
    checkpoint = torch.load(PRETRAINED_PATH, map_location=DEVICE)
    model_state_dict = model.state_dict()
    pretrained_state_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)
    print(f"Successfully loaded pre-trained weights from {PRETRAINED_PATH}.")

    # --- Loss Function, Optimizer & Scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=NUM_EPOCHS)


    # --- Training ---
    train_losses = []
    val_losses = []
    val_accuracies = []
    max_accuracy = 0.0
    start_time = time.time()

    print(f"\nStarting training for {NUM_EPOCHS} epochs [lr={LEARNING_RATE}, wd={WEIGHT_DECAY}]...")

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

        scheduler.step()

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