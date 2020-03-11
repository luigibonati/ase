from typing import Union, Dict, Any
from collections import ChainMap
from configparser import ConfigParser


class ASEConfigParser(ConfigParser):
    def _unify_values(self,
                      section: str,
                      vars: Union[Dict[str, Any], ChainMap],
                      ) -> ChainMap:
        vars_mod = ConfigParser._unify_values(self, section, vars)
        if '_' not in section:
            return vars_mod
        subsection = '_'.join(section.split('_')[:-1])
        vars_mod = self._unify_values(subsection, vars_mod)
        return ConfigParser._unify_values(self, section, vars_mod)
