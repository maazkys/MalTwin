# modules/detection/__init__.py
from .model import MalTwinCNN
from .trainer import train
from .evaluator import evaluate
from .inference import load_model, predict_single
from .gradcam import generate_gradcam, overlay_heatmap
