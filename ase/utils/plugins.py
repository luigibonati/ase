"""
Utilities for plugins to ase
"""

from typing import NamedTuple,Union,List

#Name is defined in the entry point
class ExternalIOFormat(NamedTuple):
    desc: str
    code: str
    module: str=None
    glob: Union[str,List[str]]=None
    ext: Union[str,List[str]]=None
    magic: Union[bytes,List[bytes]]=None
    magic_regex: bytes=None
    