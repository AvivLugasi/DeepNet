from Functions.Metrics.Accuracy import Accuracy
from Functions.Metrics.Consts import METRICS_FUNCS_NAMES
from Functions.Metrics.Metric import Metric


def return_metric_from_str(metric_name: str = None) -> Metric:
    if metric_name not in METRICS_FUNCS_NAMES:
        raise ValueError(f"metric_name must be one of {METRICS_FUNCS_NAMES}, got {metric_name} instead")
    else:
        if metric_name == METRICS_FUNCS_NAMES[0]:
            return Accuracy()
