def write_kpoints(parameters):
    for mode, value in parameters.items():
        mode = mode.capitalize()
        if mode == "Auto":
            num_kpoints = 0
            kpoints = value
        elif mode in ("Gamma", "Monkhorst"):
            num_kpoints = 0
            kpoints = " ".join(str(x) for x in value)
        elif mode == 'Line':
            num_kpoints = value
        elif mode in ("Reciprocal", "Cartesian"):
            kpoints = []
            for x in value:
                kpoints.append(" ".join(str(y) for y in x))
            kpoints = "\n".join(x for x in kpoints)
            mode = "Line\n" + mode
        else:
            raise NotImplementedError
    kpoints_string = f"""KPOINTS created by Atomic Simulation Environment
{num_kpoints}
{mode}
{kpoints}"""
    with open("KPOINTS", "w") as kpoints:
        kpoints.write(kpoints_string)
