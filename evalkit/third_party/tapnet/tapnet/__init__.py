"""
Lightweight tapnet init for Ref4D motion eval (torch-only).

Default: do NOT import JAX/Haiku codepath to avoid jax/haiku version issues.
If you really need the JAX pipeline, set TAPNET_ENABLE_JAX=1.
"""
import os

__all__ = ["torch"]

if os.environ.get("TAPNET_ENABLE_JAX", "0") == "1":
    # Optional JAX/Haiku path (not needed for Ref4D motion RRM)
    from tapnet.models import tapir_model  # noqa: F401
