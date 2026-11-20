import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def get_data_loaders(cfg):
    """
    Fashion-MNIST with the same preprocessing and the same
    55,000 / 5,000 train/validation split (seed 42) as Lab 3.
    Settings come from the YAML config (cfg["data"]).
    """
    d = cfg["data"]

    # Convert images to tensors and normalize them.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    full_train_dataset = datasets.FashionMNIST(
        root=d["root"], train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=d["root"], train=False, download=True, transform=transform
    )

    # 60,000 images -> 55,000 training + 5,000 validation.
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [55000, 5000],
        generator=torch.Generator().manual_seed(42)
    )

    # Optional: use fewer training images for fast experiments.
    if d.get("train_subset"):
        train_dataset = Subset(train_dataset, range(d["train_subset"]))

    nw = d["num_workers"]
    kw = dict(
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=nw > 0,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=d["batch_size"], shuffle=True, **kw
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, **kw)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, **kw)

    return train_loader, val_loader, test_loader
