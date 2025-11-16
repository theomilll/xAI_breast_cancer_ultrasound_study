"""Aplicativo Streamlit destacando a classificacao BUS_UC com XAI."""

import os
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from streamlit.watcher import local_sources_watcher as _lsw

# Add parent directory to path so src modules resolve
sys.path.append(str(Path(__file__).parent.parent))

from src.models.cls_resnet18 import ResNet18Classifier
from src.xai.grad_cam import GradCAM, GradCAMPlusPlus
from src.xai.integrated_gradients import IntegratedGradientsExplainer
from src.xai.rise import RISE


def _safe_namespace_paths(module: ModuleType) -> list[str]:
    path_attr = getattr(module, "__path__", None)
    if path_attr is None:
        return []
    raw = getattr(path_attr, "_path", None)
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return []


def _safe_get_module_paths(module: ModuleType) -> set[str]:
    paths_extractors = [
        lambda m: [getattr(m, "__file__", None)],
        lambda m: [getattr(getattr(m, "__spec__", None), "origin", None)],
        _safe_namespace_paths,
    ]
    all_paths: set[str] = set()
    for extract in paths_extractors:
        try:
            potential_paths = extract(module)
        except Exception:
            continue
        all_paths.update(
            [
                os.path.abspath(str(p))
                for p in potential_paths
                if isinstance(p, str) and (os.path.isfile(p) or os.path.isdir(p))
            ]
        )
    return all_paths


_lsw.get_module_paths = _safe_get_module_paths


IMG_SIZE = 256
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CLS_CKPT = Path("outputs/cls_resnet18/fold_0/best.ckpt")
FAITHFULNESS_PLOT = Path("outputs/xai_insertion_deletion_curves.png")
GALLERY_PLOT = Path("outputs/xai_visualization_grid.png")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


st.set_page_config(page_title="Demo BUS_UC XAI", page_icon="🔬", layout="wide")
st.title("🔬 XAI para Classificacao BUS_UC")
st.markdown("Predicoes benigna vs maligna em lote com mapas de saliencia")


st.sidebar.header("Controles de XAI")
xai_method = st.sidebar.multiselect(
    "Metodos XAI",
    ["Grad-CAM", "Grad-CAM++", "Integrated Gradients", "RISE"],
    default=["Grad-CAM"],
)
opacity = st.sidebar.slider("Opacidade do mapa de calor", 0.0, 1.0, 0.5)
region_topk = st.sidebar.slider("Percentual top-k da saliencia", 1, 40, 10)
flipbook_percent = st.sidebar.slider("Percentil do flipbook de saliencia", 5, 50, 20)
uploaded_images = st.sidebar.file_uploader(
    "Envie imagens de ultrassom",
    type=["png", "jpg", "jpeg", "bmp", "tif"],
    accept_multiple_files=True,
)


@st.cache_resource
def load_classification_model(checkpoint_path: str):
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        st.error(f"Checkpoint de classificacao nao encontrado: {ckpt}")
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


def overlay_heatmap(image, heatmap, alpha):
    cmap = plt.get_cmap("magma")
    heat = cmap(heatmap)[..., :3]
    blended = (1 - alpha) * image + alpha * heat
    return np.clip(blended, 0.0, 1.0)


def classification_visualization(image, saliency_maps, alpha, aggregate_map=None):
    overlays = list(saliency_maps.items())
    if aggregate_map is not None:
        overlays.append(("Foco medio", aggregate_map))
    cols = 1 + len(overlays)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    axes[0].imshow(image)
    axes[0].set_title("Entrada")
    axes[0].axis("off")
    for idx, (name, saliency) in enumerate(overlays, start=1):
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
            st.warning(f"{method} falhou: {err}")
    return saliency


def aggregate_saliency_map(saliency_maps: dict[str, np.ndarray]) -> np.ndarray | None:
    if not saliency_maps:
        return None
    stack = np.stack(list(saliency_maps.values()))
    agg = stack.mean(axis=0)
    agg -= agg.min()
    max_val = agg.max()
    if max_val > 0:
        agg /= max_val
    return agg


def saliency_statistics(saliency_map: np.ndarray) -> dict[str, float]:
    flat = saliency_map.flatten()
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "max": float(flat.max()),
    }


def compute_topk_mask(saliency_map: np.ndarray, percent: float) -> tuple[np.ndarray, float]:
    percent = max(0.1, min(100.0, percent))
    flat = saliency_map.flatten()
    k = max(1, int(len(flat) * percent / 100.0))
    threshold = np.partition(flat, -k)[-k]
    mask = saliency_map >= threshold
    return mask, float(threshold)


def overlay_binary_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    color = np.zeros_like(image)
    color[..., 0] = 1.0
    return np.clip((1 - alpha) * image + alpha * (color * mask[..., None]), 0.0, 1.0)


def compute_region_stats(image: np.ndarray, saliency_map: np.ndarray, percent: float) -> dict[str, float]:
    mask, threshold = compute_topk_mask(saliency_map, percent)
    coverage = float(mask.mean() * 100.0)
    if mask.any():
        mean_intensity = float(image[mask].mean())
        mean_saliency = float(saliency_map[mask].mean())
    else:
        mean_intensity = 0.0
        mean_saliency = 0.0
    return {
        "coverage": coverage,
        "mean_intensity": mean_intensity,
        "mean_saliency": mean_saliency,
        "threshold": threshold,
    }


processed_images = []

if uploaded_images:
    cls_model = load_classification_model(str(CLS_CKPT))
    if cls_model is not None:
        for image_file in uploaded_images:
            img_bytes = image_file.getvalue()
            image_tensor, image_arr = preprocess_image(BytesIO(img_bytes))
            with torch.no_grad():
                logits = cls_model(image_tensor.to(DEVICE))
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            malignant_prob = float(probs[1])
            benign_prob = float(probs[0])
            pred_label = "Maligna" if malignant_prob >= 0.5 else "Benigna"
            target_idx = 1 if malignant_prob >= 0.5 else 0
            saliency_maps = {}
            if xai_method:
                saliency_maps = generate_saliency_maps(
                    cls_model, image_tensor.to(DEVICE), xai_method, target_class=target_idx
                )
            aggregate_map = aggregate_saliency_map(saliency_maps)
            saliency_stats = {name: saliency_statistics(arr) for name, arr in saliency_maps.items()}
            avg_focus = float(
                np.mean([stats["mean"] for stats in saliency_stats.values()])
            ) if saliency_stats else 0.0
            aggregate_mean = float(aggregate_map.mean()) if aggregate_map is not None else avg_focus
            processed_images.append(
                {
                    "name": image_file.name,
                    "image_arr": image_arr,
                    "image_stats": {
                        "mean": float(image_arr.mean()),
                        "std": float(image_arr.std()),
                    },
                    "classification": {
                        "saliency": saliency_maps,
                        "aggregate": aggregate_map,
                        "saliency_stats": saliency_stats,
                        "aggregate_mean": aggregate_mean,
                        "pred_label": pred_label,
                        "malignant_prob": malignant_prob,
                        "benign_prob": benign_prob,
                        "confidence_gap": abs(malignant_prob - benign_prob),
                    },
                }
            )


if not processed_images:
    st.info("Envie pelo menos uma imagem na barra lateral para executar a inferencia.")
else:
    summary_rows = []
    for item in processed_images:
        cls_info = item["classification"]
        stats = item["image_stats"]
        summary_rows.append(
            {
                "Imagem": item["name"],
                "Predicao": cls_info["pred_label"],
                "% maligna": round(cls_info["malignant_prob"] * 100, 1),
                "% benigna": round(cls_info["benign_prob"] * 100, 1),
                "Gap de confianca (pp)": round(cls_info["confidence_gap"] * 100, 1),
                "Saliencia media": round(cls_info.get("aggregate_mean", 0.0), 3),
                "Intensidade media": round(stats["mean"], 3),
                "Desvio padrao": round(stats["std"], 3),
            }
        )

    st.subheader("Resumo em lote")
    st.dataframe(summary_rows)

    scatter_df = pd.DataFrame(
        [
            {
                "Imagem": row["Imagem"],
                "Prob maligna": row["% maligna"],
                "Saliencia media (%)": row["Saliencia media"] * 100,
                "Predicao": row["Predicao"],
                "Gap (pp)": row["Gap de confianca (pp)"],
            }
            for row in summary_rows
        ]
    )
    if not scatter_df.empty:
        st.subheader("Confianca vs foco de saliencia")
        chart = (
            alt.Chart(scatter_df)
            .mark_circle(size=120, opacity=0.8)
            .encode(
                x=alt.X("Prob maligna", title="Probabilidade maligna (%)"),
                y=alt.Y("Saliencia media (%)", title="Saliencia media (%)"),
                color="Predicao",
                size=alt.Size("Gap (pp)", title="Gap de confianca (pp)", legend=None),
                tooltip=["Imagem", "Predicao", "Prob maligna", "Saliencia media (%)", "Gap (pp)"],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Analise detalhada")
    for item in processed_images:
        st.markdown(f"#### {item['name']}")
        img_stats = item["image_stats"]
        cls_info = item["classification"]
        region_rows = []
        for method, sal_map in cls_info["saliency"].items():
            region_stat = compute_region_stats(item["image_arr"], sal_map, region_topk)
            region_rows.append(
                {
                    "Metodo": method,
                    "Cobertura %": round(region_stat["coverage"], 1),
                    "Saliencia μ %": round(region_stat["mean_saliency"] * 100, 1),
                    "Intensidade μ da regiao": round(region_stat["mean_intensity"], 3),
                }
            )
        saliency_stat_rows = [
            {
                "Metodo": method,
                "μ": round(stats["mean"], 3),
                "σ": round(stats["std"], 3),
                "max": round(stats["max"], 3),
            }
            for method, stats in cls_info.get("saliency_stats", {}).items()
        ]
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(item["image_arr"], caption="Entrada normalizada", use_column_width=True)
            st.caption(
                f"Intensidade media {img_stats['mean']:.3f} · Desvio {img_stats['std']:.3f}"
            )
            st.metric("Predicao", cls_info["pred_label"])
            st.metric("Probabilidade maligna", f"{cls_info['malignant_prob'] * 100:.1f}%")
            st.metric("Probabilidade benigna", f"{cls_info['benign_prob'] * 100:.1f}%")
            st.metric("Gap de confianca", f"{cls_info['confidence_gap'] * 100:.1f} pts")
            if saliency_stat_rows:
                st.caption("Distribuicao de saliencia (por metodo)")
                st.dataframe(saliency_stat_rows)
            if region_rows:
                st.caption(f"Estatisticas do top {region_topk}% de foco")
                st.dataframe(region_rows)
                region_df = pd.DataFrame(region_rows)
                region_long = region_df.melt(
                    id_vars="Metodo",
                    value_vars=["Cobertura %", "Saliencia μ %"],
                    var_name="Metrica",
                    value_name="Valor",
                )
                region_chart = (
                    alt.Chart(region_long)
                    .mark_bar()
                    .encode(x="Metodo", y="Valor", color="Metrica")
                    .properties(height=220)
                )
                st.altair_chart(region_chart, use_container_width=True)
        with col2:
            if cls_info["saliency"]:
                classification_visualization(
                    item["image_arr"], cls_info["saliency"], opacity, cls_info.get("aggregate")
                )
                aggregate_map = cls_info.get("aggregate")
                if aggregate_map is not None:
                    mask, threshold = compute_topk_mask(aggregate_map, flipbook_percent)
                    flip_overlay = overlay_binary_mask(item["image_arr"], mask, alpha=0.5)
                    st.image(
                        flip_overlay,
                        caption=(
                            f"Flipbook com top {flipbook_percent}% do foco (limite={threshold:.3f})"
                        ),
                        use_column_width=True,
                    )
            else:
                st.info("Selecione ao menos um metodo de XAI para visualizar a saliencia.")
        st.divider()


with st.expander("ℹ️ About"):
    st.markdown(
        """
        ### XAI para Classificacao BUS_UC

        Este visualizador executa o classificador ResNet-18 (melhor checkpoint do fold 0)
        e sobrepoe varios metodos de atribuicao para explicar decisoes benignas vs malignas.

        **O que voce pode fazer:**
        - Enviar varias imagens de ultrassom e comparar predicoes lado a lado
        - Alternar Grad-CAM, Grad-CAM++, Integrated Gradients ou RISE
        - Investigar gaps de confianca por lote e resumos de intensidade

        A segmentacao foi desativada para manter o foco na interpretabilidade da classificacao.
        """
    )

st.markdown("---")
st.markdown("Construido com PyTorch Lightning, Captum, RISE e Streamlit")

if FAITHFULNESS_PLOT.exists():
    st.subheader("Curvas de fidelidade")
    st.image(str(FAITHFULNESS_PLOT), caption="AUC de insercao/delecao", use_column_width=True)

if GALLERY_PLOT.exists():
    st.subheader("Galeria de XAI")
    st.image(str(GALLERY_PLOT), caption="Painel de comparacao entre metodos", use_column_width=True)
