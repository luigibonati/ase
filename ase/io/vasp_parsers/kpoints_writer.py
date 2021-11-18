def write_kpoints(parameters):
    num_kpoints = 0
    for mode, value in parameters.items():
        mode = mode.capitalize()
        if mode == "Auto":
            kpoints = value
        elif mode in ("Gamma", "Monkhorst"):
            kpoints = " ".join(str(x) for x in value)
        elif mode == "Line":
            num_kpoints = value
        elif mode in ("Reciprocal", "Cartesian"):
            if num_kpoints == 0:
                num_kpoints = len(value)
            else:
                mode = "Line\n" + mode
            kpoints = [" ".join(str(x) for x in kpt) for kpt in value]
            kpoints = "\n".join(x for x in kpoints)
        else:
            raise NotImplementedError
    kpoints_string = f"""KPOINTS created by Atomic Simulation Environment
{num_kpoints}
{mode}
{kpoints}"""
    with open("KPOINTS", "w") as kpoints:
        kpoints.write(kpoints_string)
