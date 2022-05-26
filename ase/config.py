import os
import configparser
from pathlib import Path

from ase.utils import lazymethod


class Config:
    @lazymethod
    def paths_and_parser(self):
        parser = configparser.ConfigParser()
        envpath = os.environ.get('ASE_CONFIG_PATH')
        if envpath is not None:
            paths = [Path(p) for p in envpath.split(':')]
        else:
            paths = [Path.home() / '.config/ase/config.ini']
        loaded_paths = parser.read(paths)
        return loaded_paths, parser

    @property
    def paths(self):
        return self.paths_and_parser()[0]

    @property
    def parser(self):
        return self.paths_and_parser()[1]


cfg = Config()
