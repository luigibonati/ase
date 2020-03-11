from configparser import ExtendedInterpolation
from .config_parser import ASEConfigParser
from .get_config import get_config_paths

config = ASEConfigParser(interpolation=ExtendedInterpolation())
config.read(get_config_paths())

__all__ = ['config']
