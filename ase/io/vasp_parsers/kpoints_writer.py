from dataclasses import dataclass


@dataclass
class _State:
    number_kpoints: int = 0
    mode: str = None
    coordinate: str = None
    specification: list = None


def write_kpoints(parameters):
    state = _State()
    for key, value in parameters.items():
        key = key.capitalize()
        if key == "Auto":
            state.mode = key
            state.specification = [str(value)]
        elif key in ("Gamma", "Monkhorst"):
            state.mode = key
            state.specification = [" ".join(str(x) for x in value)]
        elif key == "Line":
            state.mode = key
            state.number_kpoints = value
        elif key in ("Reciprocal", "Cartesian"):
            if state.number_kpoints == 0:
                state.number_kpoints = len(value)
            state.coordinate = key
            state.specification = [" ".join(str(x) for x in kpt) for kpt in value]
        else:
            raise NotImplementedError
    header = ["KPOINTS created by Atomic Simulation Environment"]
    number_kpoints = [str(state.number_kpoints)]
    mode = [] if state.mode is None else [state.mode]
    coordinate = [] if state.coordinate is None else [state.coordinate]
    kpoints_string = "\n".join(
        header + number_kpoints + mode + coordinate + state.specification
    )
    with open("KPOINTS", "w") as kpoints:
        kpoints.write(kpoints_string)
