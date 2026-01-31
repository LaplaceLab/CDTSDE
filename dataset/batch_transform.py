from typing import Any, overload, Dict, Union, List, Sequence
import random

import torch
from torch.nn import functional as F
import numpy as np

from utils.image import USMSharp, DiffJPEG, filter2D
from utils.degradation import (
    random_add_gaussian_noise_pt, random_add_poisson_noise_pt
)


class BatchTransform:
    
    @overload
    def __call__(self, batch: Any) -> Any:
        ...


class IdentityBatchTransform(BatchTransform):
    
    def __call__(self, batch: Any) -> Any:
        return batch
