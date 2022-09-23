
run_povray = True
from ase import io 
from ase.utils.ptable import ptable
import os

atoms = ptable(spacing=3)


######
atoms.write('ptable.true_colors.png')
io.write('ptable.xyz',atoms)
#########
from ase.io.pov import POVRAY
styles = list( POVRAY.material_styles_dict.keys() )

for style in styles:
    pov_name = 'ptable.%s.pov'%style
    ini_name = pov_name.replace('pov','ini')

    kwargs = { # For povray files only
        'textures': len(atoms)*[style],
        'transparent'  : True, # Transparent background
        'canvas_width' : 1000,  # Width of canvas in pixels
        'camera_type': 'orthographic angle 65', 
        }

    generic_projection_settings = {
        }

    io.write(pov_name, atoms, 
        **generic_projection_settings,
        povray_settings=kwargs)

    if run_povray:
        os.system('povray %s'%ini_name)
