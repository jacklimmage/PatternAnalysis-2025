import torch
from torchvision import datasets, transforms

DATA_DIR = "/home/groups/comp3710/ADNI/AD_NC"

def load_data(batch_size=32, num_workers=4):
    print("\nLoading data...")

    # transform to convert images to tensors
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    # load images from directory
    train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
    test_dataset  = datasets.ImageFolder(f"{DATA_DIR}/test", transform=test_transform)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # print(f"Train: {len(train_dataset)} images, Test: {len(test_dataset)} images")
    # print("Classes:", train_dataset.classes)
    return train_loader, test_loader
