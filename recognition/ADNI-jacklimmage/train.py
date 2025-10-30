import torch
from torch import nn, optim
from dataset import load_data
from modules import *
import time, datetime

BATCH_SIZE = 256
NUM_WORKERS = 4

NUM_EPOCHS = 10
STAGE_2_EPOCHS = 20
LEARNING_RATE = 5e-3
STAGE_2_LR = 5e-7
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

    # --- Freeze all parameters ---
    for param in model.parameters():
        param.requires_grad = False
    print("All model layers frozen.")

    # --- Unfreeze classification head ---
    for param in model.head.parameters():
        param.requires_grad = True
    print("Classification head unfrozen for training.")

    # --- Loss Function, Optimizer & Scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=NUM_EPOCHS)


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

        scheduler.step()

        # --- Save Best Model ---
        if val_acc_pct > max_accuracy:
            max_accuracy = val_acc_pct
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  *** Saved best model to {SAVE_PATH} with Val Acc: {max_accuracy:.2f}% ***")

    best_weights = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(best_weights)

    # 1. Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    print("All layers unfrozen.")

    # 2. Re-initialize Optimizer and Scheduler for the entire model
    # Use the tiny Stage 2 Learning Rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=STAGE_2_LR, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=STAGE_2_EPOCHS)

    # 3. Continue Training
    start_epoch = NUM_EPOCHS
    final_epochs = NUM_EPOCHS + STAGE_2_EPOCHS
    
    for epoch in range(start_epoch, final_epochs):        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, DEVICE)

        val_acc_pct = val_acc * 100
        print(f"\nEpoch {epoch+1}/{final_epochs} (Stage 2):")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc_pct:.2f}% (LR: {STAGE_2_LR})")

        scheduler.step()

        # --- Save Best Model (Across ALL Stages) ---
        if val_acc_pct > max_accuracy:
            max_accuracy = val_acc_pct
            # The SAVE_PATH variable already points to your best_model.pth
            torch.save(model.state_dict(), SAVE_PATH) 
            print(f"   *** SAVED NEW BEST MODEL to {SAVE_PATH} with Val Acc: {max_accuracy:.2f}% ***")

    # --- The End ---
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('\nTraining finished.')
    print(f'Total training time: {total_time_str}')

if __name__ == "__main__":
    main()