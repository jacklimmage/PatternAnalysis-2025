import torch
from torchvision import datasets, transforms

DATA_DIR = "/home/groups/comp3710/ADNI/AD_NC"

def load_data(batch_size=32, num_workers=4):
    print("Loading data...")

    # transform to convert images to tensors
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # load images from directory
    train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=transform)
    test_dataset  = datasets.ImageFolder(f"{DATA_DIR}/test", transform=transform)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # print(f"Train: {len(train_dataset)} images, Test: {len(test_dataset)} images")
    # print("Classes:", train_dataset.classes)
    return train_loader, test_loader
