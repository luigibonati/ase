import ase.parallel as parallel
from ase.visualize.external import viewers


def view(atoms, data=None, viewer='ase', repeat=None, block=False):
    if parallel.world.size > 1:
        return

    vwr = viewers[viewer.lower()]
    handle = vwr.view(atoms, data=data, repeat=repeat)

    if block and hasattr(handle, 'wait'):
        status = handle.wait()
        if status != 0:
            raise RuntimeError(f'Viewer "{vwr.name}" failed with status '
                               '{status}')

    return handle


def paraview_stuff():
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
