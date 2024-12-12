

import sys
from pathlib import Path
from typing import List

from detection_model import yolo  # noqa
from detection_model.nn.tasks import (DetectionModel, attempt_load_one_weight, guess_model_task)
from detection_model.yolo.cfg import get_cfg
from detection_model.yolo.utils import  callbacks
from detection_model.yolo.utils.downloads import GITHUB_ASSET_STEMS

# Map head to model, trainer, validator, and predictor classes
MODEL_MAP = {
    "detect": [
        DetectionModel, 'yolo.TYPE.detect.DetectionPredictor']}


class YOLO:
    """
    YOLO

    A python interface which emulates a model-like behaviour by wrapping trainers.
    """

    def __init__(self, model='yolov8n.pt', type="v8") -> None:
        """
        Initializes the YOLO object.

        Args:
            model (str, Path): model to load or create
            type (str): Type/version of models to use. Defaults to "v8".
        """
        self.type = type
        self.ModelClass = None  # model class
        self.PredictorClass = None  # predictor class
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.task = None  # task type
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object

        # Load YOLO model
        suffix = Path(model).suffix
        if not suffix and Path(model).stem in GITHUB_ASSET_STEMS:
            model, suffix = Path(model).with_suffix('.pt'), '.pt'  # add suffix, i.e. yolov8n -> yolov8n.pt
        try:
            self._load(model)

        except Exception as e:
            raise NotImplementedError(f"Unable to load model='{model}'. "
                                      f"As an example try model='yolov8n.pt' ") from e

    def __call__(self, source=None, stream=False, **kwargs):
        return self.predict(source, stream, **kwargs)


    def _load(self, weights: str):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
        """
        suffix = Path(weights).suffix
        if suffix == '.pt':
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args["task"]
            self.overrides = self.model.args
            self._reset_ckpt_args(self.overrides)
        else:
            self.model, self.ckpt = weights, None
            self.task = guess_model_task(weights)
        self.ckpt_path = weights
        self.ModelClass, self.PredictorClass = self._assign_ops_from_task()

    def reset(self):
        """
        Resets the model modules.
        """
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True

    def info(self, verbose=False):
        """
        Logs model info.

        Args:
            verbose (bool): Controls verbosity.
        """
        self.model.info(verbose=verbose)

    def fuse(self):
        self.model.fuse()

    def predict(self, source=None, stream=False, **kwargs):

        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[detection_model.yolo.engine.results.Results]): The prediction results.
        """
        overrides = self.overrides.copy()
        overrides["conf"] = 0.25
        overrides.update(kwargs)
        overrides["mode"] = "predict"
        overrides["save"] = kwargs.get("save", False)  # not save files by default
        if not self.predictor:
            self.predictor = self.PredictorClass(overrides=overrides)
            self.predictor.setup_model(model=self.model)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
        is_cli = sys.argv[0].endswith('yolo') or sys.argv[0].endswith('detection_model')
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def to(self, device):
        """
        Sends the model to the given device.

        Args:
            device (str): device
        """
        self.model.to(device)

    def _assign_ops_from_task(self):
        model_class,  pred_lit = MODEL_MAP[self.task]
        # warning: eval is unsafe. Use with caution
        predictor_class = eval(pred_lit.replace("TYPE", f"{self.type}"))

        return model_class,  predictor_class

    @property
    def names(self):
        """
         Returns class names of the loaded model.
        """
        return self.model.names

    @property
    def transforms(self):
        """
         Returns transform of the loaded model.
        """
        return self.model.transforms if hasattr(self.model, 'transforms') else None

    @staticmethod
    def add_callback(event: str, func):
        """
        Add callback
        """
        callbacks.default_callbacks[event].append(func)

    @staticmethod
    def _reset_ckpt_args(args):
        for arg in 'augment', 'verbose', 'project', 'name', 'exist_ok', 'resume', 'batch', 'epochs', 'cache', \
                'save_json', 'half', 'v5loader', 'device', 'cfg', 'save', 'rect', 'plots':
            args.pop(arg, None)
