"""Streamlit demo app for BUS_UC segmentation, classification, and XAI."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

# Add parent directory to path so src modules resolve
sys.path.append(str(Path(__file__).parent.parent))

from src.metrics import dice_score, iou_score
from src.models.cls_resnet18 import ResNet18Classifier
from src.models.seg_unet import UNetRes34
from src.xai.grad_cam import GradCAM, GradCAMPlusPlus
from src.xai.integrated_gradients import IntegratedGradientsExplainer
from src.xai.rise import RISE

IMG_SIZE = 256
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SEG_CKPT = Path("outputs/seg_unet/fold_0/best.ckpt")
CLS_CKPT = Path("outputs/cls_resnet18/fold_0/best.ckpt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


st.set_page_config(page_title="BUS_UC XAI Demo", page_icon="🔬", layout="wide")
st.title("🔬 Breast Ultrasound Analysis with XAI")
st.markdown("Interactive demo for lesion segmentation, classification, and explainable AI")


st.sidebar.header("Settings")
task = st.sidebar.radio("Task", ["Segmentation", "Classification"])

xai_method = []
if task == "Classification":
    xai_method = st.sidebar.multiselect(
        "XAI Methods",
        ["Grad-CAM", "Grad-CAM++", "Integrated Gradients", "RISE"],
        default=["Grad-CAM"],
    )

uploaded_image = st.sidebar.file_uploader(
    "Upload ultrasound image", type=["png", "jpg", "jpeg", "bmp", "tif"]
)
uploaded_mask = None
if task == "Segmentation":
    uploaded_mask = st.sidebar.file_uploader(
        "Optional ground-truth mask", type=["png", "jpg", "jpeg", "bmp", "tif"]
    )

show_overlay = st.sidebar.checkbox("Show overlay", value=True)
opacity = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.5)
seg_threshold = 0.5
if task == "Segmentation":
    seg_threshold = st.sidebar.slider("Segmentation threshold", 0.0, 1.0, 0.5)


@st.cache_resource
def load_segmentation_model(checkpoint_path: str):
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        st.error(f"Segmentation checkpoint not found: {ckpt}")
        return None
    model = UNetRes34(num_classes=1, in_channels=3)
    state = torch.load(ckpt, map_location="cpu")
    weights = {
        k[len("model.") :]: v for k, v in state["state_dict"].items() if k.startswith("model.")
    }
    model.load_state_dict(weights)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_classification_model(checkpoint_path: str):
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        st.error(f"Classification checkpoint not found: {ckpt}")
        return None
    model = ResNet18Classifier(num_classes=2, pretrained=False)
    state = torch.load(ckpt, map_location="cpu")
    weights = {
        k[len("model.") :]: v for k, v in state["state_dict"].items() if k.startswith("model.")
    }
    model.load_state_dict(weights)
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(upload) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(upload).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor, arr


def load_mask_tensor(upload):
    mask = Image.open(upload).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = (np.array(mask).astype(np.float32) / 255.0)
    binary = (arr > 0.5).astype(np.float32)
    tensor = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)
    return tensor, binary


def segmentation_visualization(image, prob_map, pred_mask, gt_mask=None):
    cols = 3 + (1 if gt_mask is not None else 0)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(prob_map, cmap="magma")
    axes[1].set_title("Predicted prob.")
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Thresholded mask")
    if gt_mask is not None:
        axes[3].imshow(gt_mask, cmap="gray")
        axes[3].set_title("Ground-truth mask")
    for ax in axes:
        ax.axis("off")
    st.pyplot(fig, use_container_width=True)


def overlay_heatmap(image, heatmap, alpha):
    cmap = plt.get_cmap("magma")
    heat = cmap(heatmap)[..., :3]
    blended = (1 - alpha) * image + alpha * heat
    return np.clip(blended, 0.0, 1.0)


def classification_visualization(image, saliency_maps, alpha):
    cols = 1 + len(saliency_maps)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")
    for idx, (name, saliency) in enumerate(saliency_maps.items(), start=1):
        overlay = overlay_heatmap(image, saliency, alpha)
        axes[idx].imshow(overlay)
        axes[idx].set_title(name)
        axes[idx].axis("off")
    st.pyplot(fig, use_container_width=True)


def get_cls_target_layer(model):
    return model.backbone.layer4[-1]


def generate_saliency_maps(model, input_tensor, methods, target_class):
    saliency = {}
    for method in methods:
        try:
            if method == "Grad-CAM":
                explainer = GradCAM(model, target_layer=get_cls_target_layer(model))
                saliency[method] = explainer.generate(input_tensor.clone(), target_class=target_class)
            elif method == "Grad-CAM++":
                explainer = GradCAMPlusPlus(model, target_layer=get_cls_target_layer(model))
                saliency[method] = explainer.generate(input_tensor.clone(), target_class=target_class)
            elif method == "Integrated Gradients":
                explainer = IntegratedGradientsExplainer(model, baseline="black", n_steps=50)
                saliency[method] = explainer.generate(
                    input_tensor.clone(), target_class=target_class
                )
            elif method == "RISE":
                explainer = RISE(model, input_size=(IMG_SIZE, IMG_SIZE), n_masks=400, cell_size=8)
                saliency[method] = explainer.generate(
                    input_tensor.clone(), target_class=target_class, batch_size=32
                )
        except Exception as err:  # noqa: BLE001
            st.warning(f"{method} failed: {err}")
    return saliency


image_tensor = None
image_arr = None
segmentation_result = None
seg_metrics = None
classification_result = None
pred_label = None
malignant_prob = None

if uploaded_image is not None:
    image_tensor, image_arr = preprocess_image(uploaded_image)
    if task == "Segmentation":
        seg_model = load_segmentation_model(str(SEG_CKPT))
        if seg_model is not None:
            with torch.no_grad():
                logits = seg_model(image_tensor.to(DEVICE))
                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_mask = (probs >= seg_threshold).astype(np.float32)
            gt_mask_np = None
            if uploaded_mask is not None:
                _, gt_mask_np = load_mask_tensor(uploaded_mask)
                seg_metrics = {
                    "dice": dice_score(pred_mask, gt_mask_np),
                    "iou": iou_score(pred_mask, gt_mask_np),
                }
            segmentation_result = {"prob": probs, "pred_mask": pred_mask, "gt": gt_mask_np}
    else:
        cls_model = load_classification_model(str(CLS_CKPT))
        if cls_model is not None:
            with torch.no_grad():
                logits = cls_model(image_tensor.to(DEVICE))
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            malignant_prob = float(probs[1])
            pred_label = "Malignant" if malignant_prob >= 0.5 else "Benign"
            target_idx = 1 if malignant_prob >= 0.5 else 0
            saliency_maps = {}
            if xai_method:
                saliency_maps = generate_saliency_maps(
                    cls_model, image_tensor.to(DEVICE), xai_method, target_class=target_idx
                )
            classification_result = {"saliency": saliency_maps}


col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input")
    if image_arr is None:
        st.info("Upload an image on the sidebar to run inference.")
    else:
        st.image(image_arr, caption="Normalized input", use_column_width=True)
        if task == "Segmentation":
            st.subheader("Metrics")
            if seg_metrics:
                st.metric("Dice", f"{seg_metrics['dice']:.3f}")
                st.metric("IoU", f"{seg_metrics['iou']:.3f}")
            else:
                st.info("Upload a mask to compute Dice/IoU.")
        else:
            st.subheader("Prediction")
            if malignant_prob is not None:
                st.metric("Prediction", pred_label)
                st.metric("Malignant probability", f"{malignant_prob * 100:.1f}%")
            else:
                st.warning("Classification checkpoint missing.")

with col2:
    st.subheader("Visualization")
    if task == "Segmentation":
        if segmentation_result:
            segmentation_visualization(
                image_arr,
                segmentation_result["prob"],
                segmentation_result["pred_mask"],
                segmentation_result["gt"],
            )
            if show_overlay:
                overlay = image_arr.copy()
                color = np.zeros_like(overlay)
                color[..., 0] = 1.0
                overlay = (1 - opacity) * overlay + opacity * (
                    color * segmentation_result["pred_mask"][..., None]
                )
                st.image(overlay, caption="Overlay", use_column_width=True)
        else:
            st.info("Segmentation visualization will appear here.")
    else:
        if classification_result and classification_result["saliency"]:
            classification_visualization(image_arr, classification_result["saliency"], opacity)
        elif image_arr is not None:
            st.info("Select at least one XAI method to visualize saliency.")
        else:
            st.info("Classification + XAI visualization will appear here.")


with st.expander("ℹ️ About"):
    st.markdown(
        """
        ### BUS_UC Breast Ultrasound Analysis

        This interactive demo showcases:
        - **Segmentation**: U-Net for lesion boundary detection
        - **Classification**: ResNet-18 for benign vs malignant prediction
        - **XAI**: Multiple explainability methods to visualize model decisions

        #### XAI Methods:
        - **Grad-CAM** / **Grad-CAM++**
        - **Integrated Gradients**
        - **RISE**

        #### Metrics:
        - **Segmentation**: Dice, IoU
        - **Classification**: ROC-AUC, Balanced Accuracy, calibration (offline)
        - **XAI Faithfulness**: Insertion/Deletion AUC, Pointing Game
        """
    )

st.markdown("---")
st.markdown("Built with PyTorch Lightning, segmentation_models_pytorch, Captum, and Streamlit")

