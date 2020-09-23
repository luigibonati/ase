from io import BytesIO
import os
import subprocess
import sys
import tempfile

from ase.io import write
import ase.parallel as parallel


def _pipe_to_gui(atoms, repeat, block):
    buf = BytesIO()
    write(buf, atoms, format='traj')

    args = [sys.executable, '-m', 'ase', 'gui', '-']
    if repeat:
        args.append(' --repeat={},{},{}'.format(*repeat))

    proc = subprocess.Popen(args, stdin=subprocess.PIPE)
    proc.stdin.write(buf.getvalue())
    proc.stdin.close()
    if block:
        proc.wait()
    return proc


class ExternalCommandViewer:
    def __init__(self, fmt, argv):
        self.fmt = fmt
        self.argv = argv


viewers = dict(
    vmd=ExternalCommandViewer('cube', ['vmd']),
    rasmol=ExternalCommandViewer('proteindatabank', ['rasmol', '-pdb']),
    xmakemol=ExternalCommandViewer('xyz', ['xmakemol', '-f']),
    gopenmol=ExternalCommandViewer('xyz', ['runGOpenMol']),
    avogadro=ExternalCommandViewer('cube', ['avogadro']),
)


def view(atoms, data=None, viewer='ase', repeat=None, block=False):
    # Ignore for parallel calculations:
    if parallel.world.size != 1:
        return

    vwr = viewer.lower()

    if vwr == 'ase':
        return _pipe_to_gui(atoms, repeat, block)

    if repeat is not None:
        atoms = atoms.repeat()

    if vwr == 'sage':
        from ase.visualize.sage import view_sage_jmol
        view_sage_jmol(atoms)
        return

    if vwr in ('ngl', 'nglview'):
        from ase.visualize.ngl import view_ngl
        return view_ngl(atoms)

    if vwr == 'x3d':
        from ase.visualize.x3d import view_x3d
        return view_x3d(atoms)

    if vwr in viewers:
        viewer = viewers[vwr]
        format = viewer.fmt
        command = viewer.command
    elif vwr == 'paraview':
        # macro for showing atoms in paraview
        macro = """\
from paraview.simple import *
version_major = servermanager.vtkSMProxyManager.GetVersionMajor()
source = GetActiveSource()
renderView1 = GetRenderView()
atoms = Glyph(Input=source,
              GlyphType='Sphere',
#              GlyphMode='All Points',
              Scalars='radii',
              ScaleMode='scalar',
              )
RenameSource('Atoms', atoms)
atomsDisplay = Show(atoms, renderView1)
if version_major <= 4:
    atoms.SetScaleFactor = 0.8
    atomicnumbers_PVLookupTable = GetLookupTableForArray( "atomic numbers", 1)
    atomsDisplay.ColorArrayName = ('POINT_DATA', 'atomic numbers')
    atomsDisplay.LookupTable = atomicnumbers_PVLookupTable
else:
    atoms.ScaleFactor = 0.8
    ColorBy(atomsDisplay, 'atomic numbers')
    atomsDisplay.SetScalarBarVisibility(renderView1, True)
Render()
        """
        script_name = os.path.join(tempfile.gettempdir(), 'draw_atoms.py')
        with open(script_name, 'w') as fd:
            fd.write(macro)
        format = 'vtu'
        command = 'paraview --script=' + script_name
    else:
        raise RuntimeError('Unknown viewer: ' + viewer)

    fd, filename = tempfile.mkstemp('.' + format, 'ase-')

    if data is None:
        write(filename, atoms, format=format)
    else:
        write(filename, atoms, format=format, data=data)

    viewer_args = command.split() + [filename]

    from ase.visualize.external import open_external_viewer
    viewer = open_external_viewer(viewer_args, filename)

    if block:
        status = viewer.wait()
        if status != 0:
            raise RuntimeError(
                'Viewer failed on file "{}" with status {} and args: {}'
                .format(filename, status, viewer_args))

    return viewer
