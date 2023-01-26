from pathlib import Path
from ase.db.row import row2dct
from ase.formula import Formula


class DatabaseProject:
    """Settings for web view of a database.

    For historical reasons called a "Project".
    """

    def __init__(self, name, title, *,
                 key_descriptions,
                 database,
                 default_columns):
        self.name = name
        self.title = title
        self.uid_key = 'id'
        self.key_descriptions = key_descriptions
        self.database = database
        self.default_columns = default_columns

        templates = Path('ase/db/templates')
        self.search_template = str(templates / 'search.html')
        self.row_template = str(templates / 'row.html')
        self.table_template = str(templates / 'table.html')

    def handle_query(self, args) -> str:
        """Convert request args to ase.db query string."""
        return args['query']

    def row_to_dict(self, row):
        """Convert row to dict for use in html template."""
        dct = row2dct(row, self.key_descriptions)
        dct['formula'] = Formula(row.formula).convert('abc').format('html')
        return dct

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
