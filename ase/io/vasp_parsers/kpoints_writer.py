def write_kpoints(parameters):
    with open("KPOINTS", "w") as kpoints:
        if "Auto" in parameters:
            kpoints.write(
                f"""KPOINTS created by Atomic Simulation Environment
0
Auto
{parameters['Auto']}"""
            )
        elif "Gamma" in parameters:
            kpoints.write(
                f"""KPOINTS created by Atomic Simulation Environment
0
Gamma
{" ".join(str(x) for x in parameters['Gamma'])}"""
            )
        elif "Monkhorst" in parameters:
            kpoints.write(
                f"""KPOINTS created by Atomic Simulation Environment
0
Monkhorst
{" ".join(str(x) for x in parameters['Monkhorst'])}"""
            )
