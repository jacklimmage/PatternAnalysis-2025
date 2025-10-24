import torch
from torchvision import datasets

DATA_DIR = "/home/groups/comp3710/ADNI/"

def load_data(batch_size=32):
    print("Loading data...")

    train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train")
    test_dataset  = datasets.ImageFolder(f"{DATA_DIR}/test")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)} images, Test: {len(test_dataset)} images")
    print("Classes:", train_dataset.classes)
    return train_loader, test_loader

if __name__ == "__main__":
    load_data()
    