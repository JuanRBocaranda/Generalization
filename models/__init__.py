from .simple_cnn import SimpleCNN
from .resnet import ResNet50Transfer

def get_model(config):
    """
    Factory function to instantiate the specified model.
    """
    model_type = config['model'].get('type', 'resnet50')
    num_classes = config['model'].get('num_classes', 5)
    pretrained = config['model'].get('pretrained', True)
    
    if model_type == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes)
    elif model_type == 'resnet50':
        return ResNet50Transfer(num_classes=num_classes, pretrained=pretrained, config=config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
