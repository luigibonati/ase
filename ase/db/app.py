"""WSGI Flask-app for browsing a database.

::

    +---------------------+
    | layout.html         |
    | +-----------------+ |    +--------------+
    | | search.html     | |    | layout.html  |
    | |     +           | |    | +---------+  |
    | | table.html ----------->| |row.html |  |
    | |                 | |    | +---------+  |
    | +-----------------+ |    +--------------+
    +---------------------+

You can launch Flask's local webserver like this::

    $ ase db abc.db -w

or this::

    $ python3 -m ase.db.app abc.db

"""

import io
import sys
from typing import Dict, Any, Set
from pathlib import Path

from ase.db import connect
from ase.db.core import Database
from ase.formula import Formula
from ase.db.web import create_key_descriptions, Session
from ase.db.row import row2dct, AtomsRow
from ase.db.table import all_columns


def request2string(args) -> str:
    """Converts request args to ase.db query string."""
    return args['query']


def row_to_dict(row: AtomsRow,
                project: Dict[str, Any]) -> Dict[str, Any]:
    """Convert row to dict for use in html template."""
    dct = row2dct(row, project.key_descriptions)
    dct['formula'] = Formula(Formula(row.formula).format('abc')).format('html')
    return dct


class DBApp:
    root = Path(__file__).parent.parent.parent

    def __init__(self):
        self.projects = {}

        flask = new_app(self.projects)
        self.flask = flask

        @flask.route('/')
        def frontpage():
            projectname = next(iter(self.projects))
            return flask.view_functions['search'](projectname)

    def add_project(self, name: str, db: Database) -> None:
        all_keys: Set[str] = set()
        for row in db.select(columns=['key_value_pairs'], include_data=False):
            all_keys.update(row._keys)

        key_descriptions = {key: (key, '', '') for key in all_keys}

        meta: Dict[str, Any] = db.metadata

        if 'key_descriptions' in meta:
            key_descriptions.update(meta['key_descriptions'])

        default_columns = meta.get('default_columns')
        if default_columns is None:
            default_columns = all_columns[:]

        self.projects[name] = DatabaseProject(
            name=name,
            title=meta.get('title', ''),
            key_descriptions=create_key_descriptions(key_descriptions),
            database=db,
            default_columns=default_columns)

    @classmethod
    def run_db(cls, db):
        app = cls()
        app.add_project('default', db)
        app.flask.run(host='0.0.0.0', debug=True)


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
        self.row_to_dict_function = row_to_dict
        self.handle_query_function = request2string
        self.default_columns = default_columns

        templates = Path('ase/db/templates')
        self.search_template = str(templates / 'search.html')
        self.row_template = str(templates / 'row.html')
        self.table_template = str(templates / 'table.html')

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


def new_app(projects):
    from flask import Flask, render_template, request
    app = Flask(__name__, template_folder=str(DBApp.root))

    @app.route('/<project_name>')
    @app.route('/<project_name>/')
    def search(project_name: str):
        """Search page.

        Contains input form for database query and a table result rows.
        """
        if project_name == 'favicon.ico':
            return '', 204, []  # 204: "No content"
        session = Session(project_name)
        project = projects[project_name]
        return render_template(project.search_template,
                               q=request.args.get('query', ''),
                               p=project,
                               session_id=session.id)

    @app.route('/update/<int:sid>/<what>/<x>/')
    def update(sid: int, what: str, x: str):
        """Update table of rows inside search page.

        ``what`` must be one of:

        * query: execute query in request.args (x not used)
        * limit: set number of rows to show to x
        * toggle: toggle column x
        * sort: sort after column x
        * page: show page x
        """
        session = Session.get(sid)
        project = projects[session.project_name]
        session.update(what, x, request.args, project)
        table = session.create_table(project.database,
                                     project.uid_key,
                                     keys=list(project.key_descriptions))
        return render_template(project.table_template,
                               t=table,
                               p=project,
                               s=session)

    @app.route('/<project_name>/row/<uid>')
    def row(project_name: str, uid: str):
        """Show details for one database row."""
        project = projects[project_name]
        uid_key = project.uid_key
        row = project.database.get('{uid_key}={uid}'
                                   .format(uid_key=uid_key, uid=uid))
        dct = project.row_to_dict_function(row, project)
        return render_template(project.row_template,
                               d=dct, row=row, p=project, uid=uid)

    @app.route('/atoms/<project_name>/<int:id>/<type>')
    def atoms(project_name: str, id: int, type: str):
        """Return atomic structure as cif, xyz or json."""
        row = projects[project_name].database.get(id=id)
        a = row.toatoms()
        if type == 'cif':
            b = io.BytesIO()
            a.pbc = True
            a.write(b, 'cif', wrap=False)
            return b.getvalue(), 200, []

        fd = io.StringIO()
        if type == 'xyz':
            a.write(fd, format='extxyz')
        elif type == 'json':
            con = connect(fd, type='json')
            con.write(row,
                      data=row.get('data', {}),
                      **row.get('key_value_pairs', {}))
        else:
            1 / 0

        headers = [('Content-Disposition',
                    'attachment; filename="{project_name}-{id}.{type}"'
                    .format(project_name=project_name, id=id, type=type))]
        txt = fd.getvalue()
        return txt, 200, headers

    @app.route('/gui/<int:id>')
    def gui(id: int):
        """Pop ud ase gui window."""
        from ase.visualize import view
        # XXX so broken
        arbitrary_project = next(iter(projects))
        atoms = projects[arbitrary_project].database.get_atoms(id)
        view(atoms)
        return '', 204, []

    @app.route('/test')
    def test():
        return 'hello, world!'

    @app.route('/robots.txt')
    def robots():
        return ('User-agent: *\n'
                'Disallow: /\n'
                '\n'
                'User-agent: Baiduspider\n'
                'Disallow: /\n'
                '\n'
                'User-agent: SiteCheck-sitecrawl by Siteimprove.com\n'
                'Disallow: /\n',
                200)

    return app


handle_query = request2string


def main():
    db = connect(sys.argv[1])
    DBApp.run_db(db)


if __name__ == '__main__':
    main()
