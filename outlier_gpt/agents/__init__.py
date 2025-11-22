"""Initialization for outlier_gpt.agents module.
"""
from .core import OutlierAgent

__all__ = ["OutlierAgent"]
__version__ = "0.1.0"

from . import api_client
from . import prompter
