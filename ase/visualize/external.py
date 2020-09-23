import os
from pathlib import Path
from subprocess import Popen
import sys
from typing import Sequence

def open_external_viewer(viewer_args: Sequence[str], tempfilename: str):
    return Popen([sys.executable, '-m', 'ase.visualize.external',
                  Path(tempfilename)] + list(viewer_args))


def main():
    tempfile = Path(sys.argv[1])
    if not tempfile.exists():
        raise FileNotFoundError(tempfile)

    try:
        viewer_args = sys.argv[2:]
        proc = Popen(viewer_args)
        status = proc.wait()
    finally:
        tempfile.unlink()

    raise SystemExit(status)


if __name__ == '__main__':
    main()
