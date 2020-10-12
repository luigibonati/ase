import pytest


@pytest.fixture
def calc_params_NiH():
    calc_params = {}
    calc_params["lmpcmds"] = [
        "pair_style eam/alloy",
        "pair_coeff * * NiAlH_jea.eam.alloy Ni H",
    ]
    calc_params["atom_types"] = {"Ni": 1, "H": 2}
    calc_params["log_file"] = "test.log"
    calc_params["keep_alive"] = True
    return calc_params
