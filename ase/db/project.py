from collections.abc import Mapping
from pathlib import Path
import functools
from typing import Set, Dict, Any
from ase.db.row import row2dct
from ase.formula import Formula


@functools.total_ordering
class KeyDescription:
    def __init__(self, key, shortdesc, longdesc, unit):
        self.key = key
        self.shortdesc = shortdesc
        self.longdesc = longdesc
        self.unit = unit

    # The templates like to sort key descriptions by shortdesc.
    def __eq__(self, other):
        return self.shortdesc == getattr(other, 'shortdesc', None)

    def __lt__(self, other):
        return self.shortdesc < getattr(other, 'shortdesc', self.shortdesc)


class DatabaseProject:
    """Settings for web view of a database.

    For historical reasons called a "Project".
    """
    _ase_templates = Path('ase/db/templates')

    def __init__(self, name, title, *,
                 key_descriptions,
                 database,
                 default_columns):
        self.name = name
        self.title = title
        self.uid_key = 'id'
        self.key_descriptions = {
            key: KeyDescription(key, *desc)
            for key, desc in key_descriptions.items()}
        self.database = database
        self.default_columns = default_columns

    def get_search_template(self):
        return self._ase_templates / 'search.html'

    def get_row_template(self):
        return self._ase_templates / 'row.html'

    def get_table_template(self):
        return self._ase_templates / 'table.html'

    def handle_query(self, args) -> str:
        """Convert request args to ase.db query string."""
        return args['query']

    def row_to_dict(self, row):
        """Convert row to dict for use in html template."""
        dct = row2dct(row, self.key_descriptions)
        dct['formula'] = Formula(row.formula).convert('abc').format('html')
        return dct

    def uid_to_row(self, uid):
        return self.database.get(f'{self.uid_key}={uid}')

    @classmethod
    def dummyproject(cls, **kwargs):
        _kwargs = dict(
            name='test',
            title='test',
            key_descriptions={},
            database=None,  # XXX
            default_columns=[])
        _kwargs.update(kwargs)
        return cls(**_kwargs)

    @staticmethod
    def load_db_as_ase_project(name, database):
        from ase.db.table import all_columns
        from ase.db.web import create_key_descriptions
        # If we make this a classmethod, and try to instantiate the class,
        # it would fail on subclasses.  So we use staticmethod
        all_keys: Set[str] = set()
        for row in database.select(columns=['key_value_pairs'],
                                   include_data=False):
            all_keys.update(row._keys)

        key_descriptions = {key: (key, '', '') for key in all_keys}

        meta: Dict[str, Any] = database.metadata

        if 'key_descriptions' in meta:
            key_descriptions.update(meta['key_descriptions'])

        return DatabaseProject(
            name=name,
            title=meta.get('title', ''),
            key_descriptions=create_key_descriptions(key_descriptions),
            database=database,
            default_columns=all_columns)
