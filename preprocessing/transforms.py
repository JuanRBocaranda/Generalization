import torchvision.transforms as transforms

def get_transforms(config, is_training=True):
    """
    Returns torchvision transforms for preprocessing images.
    
    Args:
        config (dict): Configuration dictionary.
        is_training (bool): Whether to include data augmentation.
        
    Returns:
        torchvision.transforms.Compose
    """
    img_size = config['preprocessing'].get('image_size', 224)
    mean = config['preprocessing'].get('normalize_mean', [0.485, 0.456, 0.406])
    std = config['preprocessing'].get('normalize_std', [0.229, 0.224, 0.225])
    
    transform_list = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    
    if is_training:
        # Optional data augmentation
        aug_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ]
        # Insert augmentation before ToTensor
        transform_list = [transform_list[0]] + aug_list + transform_list[1:]
        
    return transforms.Compose(transform_list)
