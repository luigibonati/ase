def write_kpoints(parameters):
    with open("KPOINTS", "w") as kpoints:
        kpoints.write(f"""KPOINTS created by Atomic Simulation Environment
0
Auto
{parameters['kpoints']}""")
       
