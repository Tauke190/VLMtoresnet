#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All rights reserved.
#
from .fastvit import (
    fastvit_t8,
    fastvit_t12,
    fastvit_s12,
    fastvit_sa12,
    fastvit_sa24,
    fastvit_sa36,
    fastvit_ma36,
)

from .fastvit_proposed import IMPORT_NONE
from .fastvit_proposed import fastvit_sa36_nonlocal  # noqa: F401


VANILLA_MODELS = ['fastvit_t8', 'fastvit_t12', 'fastvit_s12', 'fastvit_sa12', 'fastvit_sa24', 'fastvit_sa36', 'fastvit_ma36']
