# Análise de Ultrassom de Mama com IA Explicável (BUS_UC)

Projeto de IA médica para análise de imagens de ultrassom de mama usando o dataset BUS_UC. Implementa segmentação de lesões, classificação benigno/maligno e métodos XAI (eXplainable AI) com métricas de fidelidade.

## Visão Geral

O projeto combina três componentes principais:

- **Segmentação**: U-Net com encoder ResNet-34 para segmentar máscaras de lesões
- **Classificação**: ResNet-18 para classificar lesões como benignas ou malignas
- **XAI**: Grad-CAM, Integrated Gradients e RISE com métricas de fidelidade (Insertion/Deletion AUC, Pointing Game)

Todos os modelos são treinados com PyTorch Lightning usando validação cruzada k-fold, calibração por temperature scaling e demo interativa no Streamlit.

## Dataset BUS_UC

- **811 imagens** de ultrassom (256×256 pixels, escala de cinza)
- **358 casos benignos** (44%)
- **453 casos malignos** (56%) → desbalanceamento tratado com Focal Loss + WeightedRandomSampler
- **Máscaras**: Escala de cinza, binarizadas com `mask > 0` para treinamento
- **Splits**: Pré-computados em `data_splits.json` (3-fold ou 5-fold via config)

### Estrutura do Dataset

```
bus-uc-breast-ultrasound/
  ├─ BUS_UC/BUS_UC/BUS_UC/
  │   ├─ All/images/ + masks/          # Todos os casos
  │   ├─ Benign/images/ + masks/       # 358 benignos
  │   └─ Malignant/images/ + masks/    # 453 malignos
  └─ BUS_UC_classification/
      ├─ Benign/
      └─ Malignant/
```

## Setup Rápido

### Ambiente

```bash
conda create -n busuc python=3.11 -y
conda activate busuc
pip install -r requirements.txt
```

Ou instalação explícita:

```bash
conda create -n busuc python=3.11 -y
conda activate busuc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install lightning==2.4.0 albumentations==1.4.11 opencv-python==4.10.0.84
pip install numpy scipy scikit-image scikit-learn pandas matplotlib pillow tqdm
pip install captum==0.7.0 streamlit==1.38.0 pyyaml==6.0.2 segmentation-models-pytorch
```

### Verificação

```bash
make setup  # Verifica instalação do PyTorch + Lightning
```

## Uso

### Treinar Modelos

```bash
# Segmentação (3-fold CV, U-Net)
make train CFG=configs/seg_unet.yaml SEED=42

# Classificação (3-fold CV, ResNet-18)
make train CFG=configs/cls_resnet18.yaml SEED=42

# Direto via Python
python -m src.train --config configs/seg_unet.yaml --seed 42
```

### Demo Interativa

```bash
streamlit run streamlit_app/app.py
```

Interface web com:
- Upload de imagens de ultrassom
- Segmentação + classificação em tempo real
- Visualização XAI (Grad-CAM, IG, RISE) sobreposta às predições

### Outros Comandos

```bash
make lint   # Ruff linting em src/
make clean  # Limpa caches (__pycache__, .ipynb_checkpoints)
```

## Arquitetura

### Segmentação

- **Modelo**: U-Net com encoder ResNet-34 (pré-treinado ImageNet)
- **Loss**: 0.5·Dice + 0.5·BCE (ou Focal-Tversky para lesões pequenas)
- **Métricas**: Dice, IoU, Boundary F1/IoU

### Classificação

- **Modelo**: ResNet-18 (pré-treinado ImageNet)
- **Loss**: Focal Loss (γ=1.5) com pesos de classe automáticos
- **Métricas**: ROC-AUC, acurácia balanceada, sensibilidade, especificidade, ECE
- **Calibração**: Temperature scaling aplicado no conjunto de validação

### XAI (Explainable AI)

Três métodos implementados:

1. **Grad-CAM / Grad-CAM++**: Saliência baseada em gradientes do último bloco convolucional
2. **Integrated Gradients**: Baseline zero/média, 50 steps, opcional SmoothGrad
3. **RISE**: N=400-800 máscaras aleatórias, saliência esperada (método black-box)

**Métricas de Fidelidade** (`src/xai/faithfulness.py`):
- **Insertion/Deletion AUC**: Revelar/ocultar progressivamente regiões de alta saliência
- **Pointing Game**: Pico de saliência dentro da máscara GT?
- **Saliency IoU**: Binarizar top-k% saliência vs máscara
- **Sanity Check**: Randomizar pesos → saliência deve degradar

## Hiperparâmetros

### Segmentação (`configs/seg_unet.yaml`)

```yaml
model: U-Net (ResNet-34 encoder)
loss: Dice+BCE (0.5 cada)
optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
scheduler: CosineAnnealingLR
batch_size: 16
epochs: 100
early_stopping: patience=10 (Dice)
amp: true
k_folds: 3
```

### Classificação (`configs/cls_resnet18.yaml`)

```yaml
model: ResNet-18
loss: Focal (γ=1.5, α=auto)
optimizer: AdamW (lr=5e-4, weight_decay=1e-4)
scheduler: OneCycleLR
batch_size: 32
epochs: 80
early_stopping: patience=10 (ROC-AUC)
amp: true
k_folds: 3
```

## Resultados

### Quality Gates (antes de promover modelo)

**Segmentação**:
- Dice médio ≥ 0.80
- Boundary F1 ≥ 0.70
- Nenhum fold < 0.70

**Classificação**:
- ROC-AUC ≥ 0.90
- ECE ≤ 0.05 (pós-calibração)

**XAI**:
- Insertion AUC alto, Deletion AUC baixo
- Pointing Game ≥ 0.8 em casos malignos

### Resultados Atuais (3-fold CV)

Resultados agregados salvos em:
- `/outputs/seg_unet/cv_results.json`
- `/outputs/cls_resnet18/cv_results.json`

Visualizações XAI:
- `/outputs/xai_visualization_grid.png` (painel: imagem→seg→Grad-CAM→IG→RISE)
- `/outputs/xai_insertion_deletion_curves.png` (curvas de fidelidade)

## Estrutura de Diretórios

```
xai/
├─ bus-uc-breast-ultrasound/      # Subprojeto com dados
│   ├─ BUS_UC/                     # Dataset (811 imagens + máscaras)
│   ├─ src/
│   │   ├─ datamodules/            # Lightning DataModules
│   │   ├─ models/                 # seg_unet.py, cls_resnet18.py
│   │   ├─ transforms/             # Albumentations + speckle aug
│   │   ├─ xai/                    # grad_cam, IG, RISE, faithfulness
│   │   ├─ train.py                # Loop de treinamento Lightning
│   │   ├─ losses.py               # Dice+BCE, Focal, Tversky
│   │   └─ metrics.py              # Dice, IoU, Boundary F1, ROC-AUC, ECE
│   ├─ configs/                    # seg_unet.yaml, cls_resnet18.yaml
│   ├─ notebooks/                  # 00_eda, 10_train_seg, 30_xai_eval
│   ├─ streamlit_app/app.py        # Demo interativa
│   ├─ outputs/                    # Checkpoints, métricas, visualizações
│   ├─ data_splits.json            # Splits k-fold pré-computados
│   ├─ requirements.txt
│   └─ Makefile
└─ README.md                       # Este arquivo
```

## Augmentação de Dados

Pipeline Albumentations (pareado para segmentação):

- `HorizontalFlip(p=0.5)`
- `Rotate(±15°, p=0.5)`
- `ElasticTransform(p=0.3)`
- `RandomBrightnessContrast(0.1, p=0.3)`
- **Speckle**: `img * (1 + s*rayleigh_noise)` (ruído característico de ultrassom)

Val/test: apenas resize + normalização determinística.

## Reprodutibilidade

- **Seed fixo**: 42 em todo lugar (dataset, dataloader workers, PyTorch/Lightning)
- **Snapshot**: Config YAML + commit hash do git salvos junto com checkpoints
- **MODEL_CARD.md**: Estatísticas de dados, métricas, calibração, resultados XAI

## Armadilhas Comuns

1. **Máscaras não-binárias**: Sempre binarizar para treino (`mask > 0`)
2. **Overfitting** (N=811): Usar weight decay, early stopping, augmentação forte, backbones pequenos
3. **Fragilidade de saliência**: Incluir sanity checks, usar RISE (black-box) para validar métodos baseados em gradiente
4. **Desbalanceamento**: Splits estratificados, Focal Loss, reportar acurácia balanceada

## Notebooks

- `notebooks/00_eda.ipynb`: Validação de dados, análise exploratória, splits estratificados
- `notebooks/10_train_segmentation.ipynb`: Guia de treinamento de segmentação
- `notebooks/30_xai_evaluation.ipynb`: Métricas XAI + visualizações

## Citação

Se usar este código, cite o dataset BUS_UC:

```
Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A.
Dataset of breast ultrasound images.
Data in Brief. 2020 Feb;28:104863.
```

## Licença

Projeto acadêmico - Universidade. Ver políticas da instituição para uso e distribuição.
