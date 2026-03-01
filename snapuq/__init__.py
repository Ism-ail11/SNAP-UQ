__version__ = "0.1.0"

from .snapuq import SnapUQ, SnapUQOutputs, GaussianPredictor, LogisticCalibrator  # noqa: F401
from .train import TrainConfig, train_classifier, train_snapuq_posthoc  # noqa: F401
from .eval import EvalConfig, collect_scores, eval_ood_detection, eval_selective_risk  # noqa: F401
from .metrics import auprc, auroc, risk_coverage_curve, detection_delay  # noqa: F401
from .calibration import fit_logistic_calibrator, fit_isotonic_calibrator  # noqa: F401

__all__ = [
    "__version__",
    "SnapUQ", "SnapUQOutputs", "GaussianPredictor", "LogisticCalibrator",
    "TrainConfig", "train_classifier", "train_snapuq_posthoc",
    "EvalConfig", "collect_scores", "eval_ood_detection", "eval_selective_risk",
    "auprc", "auroc", "risk_coverage_curve", "detection_delay",
    "fit_logistic_calibrator", "fit_isotonic_calibrator",
]
