.PHONY: setup lint train clean

SEED ?= 42
CFG ?= configs/seg_unet.yaml

setup:
	@echo "Checking PyTorch installation..."
	@python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
	@echo "Checking Lightning..."
	@python -c "import lightning; print(f'Lightning version: {lightning.__version__}')"

lint:
	@echo "Running ruff..."
	@python -m pip install -q ruff
	@ruff check src/ notebooks/ streamlit_app/

train:
	@echo "Training with config: $(CFG), seed: $(SEED)"
	@python -m src.train --config $(CFG) --seed $(SEED)

clean:
	@echo "Cleaning cache and temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
