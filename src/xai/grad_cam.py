"""Grad-CAM implementation for segmentation and classification models.

For segmentation, computes saliency w.r.t. predicted foreground mask.
For classification, computes saliency w.r.t. target class logits.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable


class GradCAM:
    """Gradient-weighted Class Activation Mapping for segmentation.

    Computes which input regions contribute most to the predicted mask.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: Segmentation model (returns logits)
            target_layer: Layer to compute CAM from (e.g., last decoder conv)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_mask: Optional target mask for segmentation (mutually exclusive with target_class)
            target_class: Optional target class for classification (mutually exclusive with target_mask)
            normalize: Normalize output to [0, 1]

        Returns:
            CAM heatmap (H, W) in range [0, 1] if normalized
        """
        self.model.eval()
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        input_tensor.requires_grad = True

        # Forward pass
        logits = self.model(input_tensor)

        # Compute target score
        if target_class is not None:
            # Classification: score w.r.t. target class logit
            # logits shape: (B, num_classes)
            probs = F.softmax(logits, dim=1)
            target_score = probs[:, target_class].sum()
        elif target_mask is not None:
            # Segmentation: Use provided GT mask as target
            # logits shape: (B, 1, H, W)
            target_mask = target_mask.to(logits.device)
            target_score = (logits * target_mask).sum()
        else:
            # Segmentation: Use predicted foreground as target
            # logits shape: (B, 1, H, W)
            probs = torch.sigmoid(logits)
            target_score = probs.sum()

        # Backward pass
        self.model.zero_grad()
        target_score.backward()

        # Compute Grad-CAM
        gradients = self.gradients  # (1, C, H', W')
        activations = self.activations  # (1, C, H', W')

        # Global average pooling of gradients (weights)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H', W')

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Convert to numpy
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if normalize and cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


class GradCAMPlusPlus(GradCAM):
    """Improved Grad-CAM with better localization for multiple instances.

    Uses weighted combination of gradients instead of simple averaging.
    """

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap.

        Uses alpha-weighted gradients for better multi-instance localization.
        """
        self.model.eval()
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        input_tensor.requires_grad = True

        # Forward pass
        logits = self.model(input_tensor)

        # Compute target score
        if target_class is not None:
            # Classification: score w.r.t. target class
            probs = F.softmax(logits, dim=1)
            target_score = probs[:, target_class].sum()
        elif target_mask is not None:
            # Segmentation: use provided GT mask
            target_mask = target_mask.to(logits.device)
            target_score = (logits * target_mask).sum()
        else:
            # Segmentation: use predicted foreground
            probs = torch.sigmoid(logits)
            target_score = probs.sum()

        # Backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients  # (1, C, H', W')
        activations = self.activations  # (1, C, H', W')

        # Compute alpha weights (Grad-CAM++ formulation)
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)

        # Avoid division by zero
        alpha = grad_2 / (2 * grad_2 + (grad_3 * activations).sum(dim=(2, 3), keepdim=True) + 1e-8)

        # ReLU on gradients before weighting
        relu_grad = F.relu(gradients)

        # Weighted pooling
        weights = (alpha * relu_grad).sum(dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Convert to numpy
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        if normalize and cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


def get_target_layer(model: torch.nn.Module, layer_name: str = "decoder") -> torch.nn.Module:
    """Helper to extract target layer from U-Net model.

    Args:
        model: Segmentation model
        layer_name: Which layer to target ("decoder", "encoder_layer4", etc.)

    Returns:
        Target layer module
    """
    # For segmentation_models_pytorch U-Net
    if hasattr(model, "model"):
        base_model = model.model
    else:
        base_model = model

    if layer_name == "decoder":
        # Last decoder block
        if hasattr(base_model, "decoder"):
            return base_model.decoder.blocks[-1]
        elif hasattr(base_model, "segmentation_head"):
            # Fallback: use layer before segmentation head
            return list(base_model.children())[-2]
    elif layer_name.startswith("encoder"):
        # Encoder layer (e.g., "encoder_layer4")
        if hasattr(base_model, "encoder"):
            layer_idx = int(layer_name.split("_")[-1].replace("layer", ""))
            return base_model.encoder.layer[layer_idx]

    raise ValueError(f"Could not find layer: {layer_name}")


def get_cls_target_layer(model: torch.nn.Module, layer_name: str = "layer4") -> torch.nn.Module:
    """Helper to extract target layer from ResNet-based classification model.

    Args:
        model: Classification model (e.g., ResNet-18)
        layer_name: Which layer to target ("layer4", "layer3", etc.)

    Returns:
        Target layer module
    """
    # Handle Lightning wrapper
    if hasattr(model, "model"):
        base_model = model.model
    else:
        base_model = model

    # Navigate ResNet backbone
    if hasattr(base_model, "backbone"):
        # Custom ResNet wrapper (e.g., ResNet18Classifier)
        backbone = base_model.backbone
    elif hasattr(base_model, "layer4"):
        # Direct torchvision ResNet
        backbone = base_model
    else:
        raise ValueError("Model structure not recognized for classification CAM")

    # Get target layer
    if layer_name == "layer4":
        return backbone.layer4[-1]  # Last BasicBlock/Bottleneck in layer4
    elif layer_name == "layer3":
        return backbone.layer3[-1]
    elif layer_name == "layer2":
        return backbone.layer2[-1]
    elif layer_name == "layer1":
        return backbone.layer1[-1]
    else:
        raise ValueError(f"Unknown layer name: {layer_name}")


def compute_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer_name: str = "decoder",
    use_plusplus: bool = False,
    target_mask: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
    task: str = "segmentation",
) -> np.ndarray:
    """Convenience function to compute Grad-CAM in one call.

    Args:
        model: Segmentation or classification model
        input_tensor: Input image (1, C, H, W)
        target_layer_name: Layer name to compute CAM from
        use_plusplus: Use Grad-CAM++ instead of standard Grad-CAM
        target_mask: Optional GT mask (segmentation only)
        target_class: Optional target class (classification only)
        task: "segmentation" or "classification"

    Returns:
        Saliency map (H, W) normalized to [0, 1]
    """
    if task == "classification":
        target_layer = get_cls_target_layer(model, target_layer_name)
    else:
        target_layer = get_target_layer(model, target_layer_name)

    if use_plusplus:
        cam_computer = GradCAMPlusPlus(model, target_layer)
    else:
        cam_computer = GradCAM(model, target_layer)

    return cam_computer.generate(input_tensor, target_mask=target_mask, target_class=target_class)
