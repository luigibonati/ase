import os
import sys
from pathlib import Path


def get_config_paths():
    paths = [Path(__file__).parent / 'defaults.conf']
    if sys.platform == 'win32':
        paths += _get_windows_config_paths()
    elif sys.platform == 'darwin':
        paths += _get_macos_config_paths()
    else:
        paths += _get_nix_config_paths()

    return [path for path in paths if path.is_file()]


def _get_windows_config_paths():
    import winreg
    key = winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders',
    )
    paths = []
    for datadir in ('AppData', 'Common AppData', 'Local AppData'):
        paths.append(Path(winreg.QueryValueEx(key, datadir)[0]) / 'ase.conf')
    return paths


def _get_macos_config_paths():
    return [Path('/Library/Preferences/ase.conf'),
            Path().home() / '/Library/Preferences/ase.conf']


def _get_nix_config_paths():
    xdgconfig = os.getenv('XDG_CONFIG_DIRS', '/etc')
    paths = [Path(path) / 'ase.conf' for path in xdgconfig.split(':')]
    xdghome = os.getenv('XDG_CONFIG_HOME', str(Path.home() / '.config'))
    paths += [Path(path) / 'ase.conf' for path in xdghome.split(':')]
    return paths
