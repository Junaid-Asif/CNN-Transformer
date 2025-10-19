import torchvision.transforms as T

def build_transforms(image_size=224, mean=None, std=None, augment_cfg=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    augment_cfg = augment_cfg or {}

    train_transforms = [T.Resize((image_size, image_size))]

    if augment_cfg.get("jitter", False):
        train_transforms.append(T.ColorJitter(
            brightness=augment_cfg.get("brightness", 0.0),
            contrast=augment_cfg.get("contrast", 0.0),
        ))

    if augment_cfg.get("horizontal_flip", False):
        train_transforms.append(T.RandomHorizontalFlip(p=0.5))

    if augment_cfg.get("vertical_flip", False):
        train_transforms.append(T.RandomVerticalFlip(p=0.5))

    if augment_cfg.get("rotation_deg", 0) > 0:
        train_transforms.append(T.RandomRotation(degrees=augment_cfg["rotation_deg"]))

    train_transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    if augment_cfg.get("random_erasing", False):
        train_transforms.append(T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)))

    val_test_transforms = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return T.Compose(train_transforms), val_test_transforms, val_test_transforms
