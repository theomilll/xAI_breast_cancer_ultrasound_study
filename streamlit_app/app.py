"""Streamlit demo app for BUS_UC segmentation, classification, and XAI."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.seg_unet import UNetRes34
from src.models.cls_resnet18 import ResNet18Classifier
from src.xai.grad_cam import GradCAM
from src.xai.integrated_gradients import IntegratedGradientsExplainer
from src.xai.rise import RISE
from src.calibration import ModelWithTemperature
from src.metrics import dice_score, iou_score


# Page config
st.set_page_config(
    page_title="BUS_UC XAI Demo",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Breast Ultrasound Analysis with XAI")
st.markdown("Interactive demo for lesion segmentation, classification, and explainable AI")


# Sidebar controls
st.sidebar.header("Settings")

task = st.sidebar.radio("Task", ["Segmentation", "Classification"])

# TODO: Add model selection dropdown
# model_checkpoint = st.sidebar.selectbox("Model Checkpoint", [...])

# TODO: Add test image selection
# test_image = st.sidebar.selectbox("Test Image", [...])

# XAI method selection
if task == "Classification":
    xai_method = st.sidebar.multiselect(
        "XAI Methods",
        ["Grad-CAM", "Grad-CAM++", "Integrated Gradients", "RISE"],
        default=["Grad-CAM"]
    )

# Visualization options
show_overlay = st.sidebar.checkbox("Show overlay", value=True)
opacity = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.5)

if task == "Segmentation":
    seg_threshold = st.sidebar.slider("Segmentation threshold", 0.0, 1.0, 0.5)
else:
    use_calibration = st.sidebar.checkbox("Use calibrated probabilities", value=True)


# Load models
@st.cache_resource
def load_segmentation_model(checkpoint_path):
    """Load trained segmentation model."""
    # TODO: Implement model loading
    # model = UNetRes34.load_from_checkpoint(checkpoint_path)
    # model.eval()
    # return model
    st.warning("Model loading not yet implemented")
    return None


@st.cache_resource
def load_classification_model(checkpoint_path, calibrated=False):
    """Load trained classification model with optional calibration."""
    # TODO: Implement model loading
    # model = ResNet18Classifier.load_from_checkpoint(checkpoint_path)
    # if calibrated:
    #     model = ModelWithTemperature(model)
    #     model.load_temperature(...)
    # model.eval()
    # return model
    st.warning("Model loading not yet implemented")
    return None


def load_test_image(image_path):
    """Load and preprocess test image."""
    # TODO: Implement image loading and preprocessing
    # - Load image
    # - Resize to 256×256
    # - Normalize
    # - Convert to tensor
    pass


def generate_saliency_map(model, image, method="Grad-CAM"):
    """Generate XAI saliency map."""
    # TODO: Implement saliency generation
    # if method == "Grad-CAM":
    #     explainer = GradCAM(model, target_layer=...)
    # elif method == "Integrated Gradients":
    #     explainer = IntegratedGradientsExplainer(model)
    # elif method == "RISE":
    #     explainer = RISE(model)
    # return explainer.generate(image)
    pass


def visualize_segmentation(image, mask, pred_mask, threshold=0.5):
    """Create segmentation visualization panel."""
    # TODO: Implement visualization
    # - Create subplot grid
    # - Show: input | GT mask | predicted mask | overlay
    # - Compute and display Dice, IoU
    pass


def visualize_classification(image, label, pred_prob, saliency_maps):
    """Create classification + XAI visualization panel."""
    # TODO: Implement visualization
    # - Show input image
    # - Display prediction probability
    # - Overlay each saliency map
    # - Create grid layout
    pass


# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input")
    # TODO: Display selected test image
    st.info("Select an image from the sidebar")

    if task == "Segmentation":
        st.subheader("Metrics")
        # TODO: Display segmentation metrics
        # - Dice score
        # - IoU
        # - Boundary F1
        pass
    else:
        st.subheader("Prediction")
        # TODO: Display classification results
        # - Predicted class
        # - Probability (calibrated/uncalibrated)
        # - Confidence
        pass

with col2:
    st.subheader("Visualization")

    if task == "Segmentation":
        st.info("Segmentation visualization will appear here")
        # TODO: Call visualize_segmentation()

    else:
        st.info("Classification + XAI visualization will appear here")
        # TODO: Call visualize_classification()


# Additional info section
with st.expander("ℹ️ About"):
    st.markdown("""
    ### BUS_UC Breast Ultrasound Analysis

    This interactive demo showcases:
    - **Segmentation**: U-Net for lesion boundary detection
    - **Classification**: ResNet-18 for benign vs malignant prediction
    - **XAI**: Multiple explainability methods to visualize model decisions

    #### XAI Methods:
    - **Grad-CAM**: Gradient-weighted Class Activation Mapping
    - **Grad-CAM++**: Improved localization for multiple instances
    - **Integrated Gradients**: Attribution by path integration
    - **RISE**: Black-box saliency via random masking

    #### Metrics:
    - **Segmentation**: Dice, IoU, Boundary F1
    - **Classification**: ROC-AUC, Balanced Accuracy, ECE (calibration)
    - **XAI Faithfulness**: Insertion/Deletion AUC, Pointing Game
    """)


# Footer
st.markdown("---")
st.markdown("Built with PyTorch Lightning, segmentation_models_pytorch, Captum, and Streamlit")


# TODO: Add additional features
# - Side-by-side comparison mode
# - Export functionality (save visualizations)
# - Batch processing mode
# - Performance metrics dashboard
# - Uncertainty visualization for predictions
