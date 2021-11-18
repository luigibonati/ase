def write_kpoints(parameters):
    for mode, value in parameters.items():
        mode = mode.capitalize()
        if mode == "Auto":
            kpoints = value
        elif mode in ("Gamma", "Monkhorst"):
            kpoints = " ".join(str(x) for x in value)
        else:
            raise NotImplementedError
    kpoints_string = f"""KPOINTS created by Atomic Simulation Environment
0
{mode}
{kpoints}"""
    with open("KPOINTS", "w") as kpoints:
        kpoints.write(kpoints_string)
