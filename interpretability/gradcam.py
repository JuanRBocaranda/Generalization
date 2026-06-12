import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt

class GradCAM:
    """
    Computes Grad-CAM visualizations.
    Reference: https://arxiv.org/abs/1610.02391
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        # Depending on PyTorch version, grad_output is a tuple.
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        """
        Generate Grad-CAM for a given input tensor.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (1, C, H, W)
            class_idx (int, optional): The class index we want to visualize. 
                                       If None, uses the model's top prediction.
                                       
        Returns:
            np.array: Heatmap overlay function.
        """
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Target for backprop
        score = output[:, class_idx]
        
        # Zero gradients and backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # GAP on gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of forward activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        return cam.squeeze().cpu().detach().numpy()

def save_gradcam(image_tensor, heatmap, save_path, original_image=None):
    """
    Overlays the heatmap on the original image and saves it.
    
    Args:
        image_tensor (torch.Tensor): The normalized input tensor (C, H, W)
        heatmap (np.array): Grad-CAM heatmap (H', W')
        save_path (str): File path to save the combined image.
        original_image (np.array, optional): If provided, overlays on the 
            denormalized original image instead. Expected shape (H, W, C) range [0, 255].
    """
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = np.float32(heatmap_colored) / 255
    
    if original_image is None:
        # Dummy denormalize for visualization (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        
        img = image_tensor.cpu().numpy()
        img = std * img + mean
        img = np.clip(img, 0, 1)
        img = np.transpose(img, (1, 2, 0)) # To (H, W, C)
    else:
        img = np.float32(original_image) / 255
        
    cam = heatmap_colored + img
    cam = cam / np.max(cam)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, np.uint8(255 * cam))

def get_target_layer(model, config):
    """
    Selects the correct target layer for Grad-CAM depending on the config.
    """
    if config.get('model', {}).get('use_cbam', False):
        # Target the conv layer inside SpatialAttention of the last CBAM block in layer4
        return model.base_model.layer4[-1].cbam.sa.conv
    else:
        # Fallback to the default last convolutional layer in ResNet50
        return model.base_model.layer4[-1].conv3

