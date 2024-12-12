
__version__ = "8.0.35"

from detection_model.yolo.engine.model import YOLO
from detection_model.yolo.utils.checks import check_yolo as checks

__all__ = ["__version__", "YOLO",  "checks"]  # allow simpler import
