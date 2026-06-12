import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import types
from models.cbam import CBAM

def add_cbam_to_resnet(model, layers=['layer2', 'layer3', 'layer4'], reduction=16):
    """
    Inyecta el módulo CBAM en los bloques bottleneck de las capas especificadas.
    """
    for layer_name in layers:
        layer = getattr(model, layer_name, None)
        if layer is None:
            continue
            
        for name, bottleneck in layer.named_children():
            # Crear CBAM y añadirlo al bottleneck
            bottleneck.cbam = CBAM(bottleneck.conv3.out_channels, reduction=reduction)
            
            # Inicialización con xavier_uniform_
            for m in bottleneck.cbam.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    if hasattr(m, 'weight') and m.weight is not None:
                        nn.init.xavier_uniform_(m.weight)
            
            # Definir el nuevo forward
            def new_forward(self, x):
                identity = x
                
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)
                
                out = self.conv3(out)
                out = self.bn3(out)
                
                # Insertar CBAM después de bn3 y antes de sumar el skip connection
                out = self.cbam(out)
                
                if self.downsample is not None:
                    identity = self.downsample(x)
                    
                out += identity
                out = self.relu(out)
                
                return out
                
            # Bind del nuevo método a la instancia
            bottleneck.forward = types.MethodType(new_forward, bottleneck)
            
    return model

class ResNet50Transfer(nn.Module):
    """
    ResNet50 model adapted for Transfer Learning on DR detection.
    """
    def __init__(self, num_classes=5, pretrained=True, config=None):
        super(ResNet50Transfer, self).__init__()
        
        # Load pre-trained ResNet50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.base_model = resnet50(weights=weights)
        
        if config and config.get('model', {}).get('use_cbam', False):
            layers_to_add = config['model'].get('cbam_layers', ['layer2', 'layer3', 'layer4'])
            reduction = config['model'].get('cbam_reduction', 16)
            self.base_model = add_cbam_to_resnet(self.base_model, layers=layers_to_add, reduction=reduction)
        
        # We extract the number of features of the last layer
        num_ftrs = self.base_model.fc.in_features
        
        # Replace the classifier layer with a new one for our num_classes
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.base_model(x)
