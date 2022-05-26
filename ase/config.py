import os
import configparser
from pathlib import Path

from ase.utils import lazyproperty


class Config:
    @lazyproperty
    def parser(self):
        parser = configparser.ConfigParser()
        envpath = os.environ.get('ASE_CONFIG_PATH')
        if envpath is not None:
            paths = [Path(p) for p in envpath.split(':')]
        else:
            paths = [Path.home() / '.config/ase/config.ini']
        parser.read(paths)
        return parser


cfg = Config()
