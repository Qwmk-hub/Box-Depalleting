import attrs
import numpy as np


@attrs.define(kw_only=True)
class InstanceData:
    bbox: np.ndarray = None
    label: str = None
    score: float = None
    mask: np.ndarray = None
