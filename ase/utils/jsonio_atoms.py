from typing import Dict, Any
import copy

""" These functions could not be in ase.io.jsonio since
    a circular dependency arises. """

def convert_to_ase_json_dict(dct: Dict[str, Any]) -> Dict[str, Any]:
    """ ... to preserve types of keys in a dictionary.
        It makes the data json-safe.
    """
    return {'__atoms_info_type__': 'dict',
            'keys': list(dct),
            'values': list(dct.values())}


def convert_to_dict(dct: Dict[str, Any]) -> Dict[str, Any]:
    """ ... convert back to original dictionary from ASE's
        json-safe dictionary. """
    return dict(zip(dct['keys'], dct['values']))


def get_json_safe_atoms_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """ A dictionary atoms.info['occupancy'] should have integer keys.
    Default JSON machinery converts all dictionary keys to strings.
    Hence, here is a correcting procedure. The procedure converts
    every dictionary (in atoms.info) to a json-safe format.
    The procedure is not recursive.
    """
    info_json = copy.deepcopy(info)
    for k, v in info_json.items():
        if isinstance(v, dict):
            dct_temp = convert_to_ase_json_dict(v)
            v.clear()
            v.update(dct_temp)
    return info_json


def get_original_atoms_info(info_json: Dict[str, Any]) -> Dict[str, Any]:
    """ A dictionary info['occupancy'] should have integer keys.
    Default JSON machinery converts all dictionary keys to strings.
    Hence, here is a procedure converting all dictionaries from
    ASE's json-safe to original dictionaries.
    """
    info = copy.deepcopy(info_json)
    for k, v in info.items():
        if isinstance(v, dict):
            atoms_info_type = v.get('__atoms_info_type__', None)
            if atoms_info_type == 'dict':
                dct_temp = convert_to_dict(v)
                v.clear()
                v.update(dct_temp)
    return info
