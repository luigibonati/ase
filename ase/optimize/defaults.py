"""Default parameters for optimizer class - currently just sets max step size"""
import collections

_bfgs = "bfgs"
_dct = {"alpha": 70.0}
bfgs_defaults = collections.namedtuple(_bfgs,
                                       _dct.keys())(**_dct)  # type: ignore

_mdmin = "mdmin"
_dct = {"dt": 0.2}
mdmin_defaults = collections.namedtuple(_mdmin,
                                        _dct.keys())(**_dct)  # type: ignore


_dct = {
    "maxstep": 0.2,  # default maxstep for all optimizers
    _bfgs: bfgs_defaults,  # type: ignore
    _mdmin: mdmin_defaults,  # type: ignore
}
defaults = collections.namedtuple("optimizer_defaults",
                                  _dct.keys())(**_dct)  # type: ignore
