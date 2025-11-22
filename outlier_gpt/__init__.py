"""Top-level package for outlier_gpt."""

__author__ = """Devminda Abeynayake"""
__email__ = "devmindaabeynayake@gmail.com"

from .agents import OutlierAgent

__version__ = "0.1.0"
__all__ = ["OutlierAgent"]

from . import techniques
from . import agents
