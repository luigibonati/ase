from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, Tuple, Union

from ase.db.core import Database, QueryParameters

ValueType = Union[int, float, bool, str]


@contextmanager
def no_progress_bar(iterable: Iterable,
                    length: int = None) -> Iterator[Iterable]:
    """A do-nothing implementation."""
    yield iterable


def block(db: Database,
          query_parameters: QueryParameters,
          blocksize: int):
    offset = query_parameters.offset
    limit = query_parameters.limit

    if limit == -1:
        limit = 99999999999999

    while True:
        blocksize = min(limit, blocksize)
        n = 0
        for row in db.select(query_parameters.query,
                             sort=query_parameters.sort,
                             limit=blocksize,
                             offset=offset):
            yield row
            n += 1
        if n < blocksize:
            return
        limit -= n
        if limit == 0:
            return
        offset += n


def insert_into(*,
                source: Database,
                destination: Database,
                query_parameters: QueryParameters,
                add_key_value_pairs: Dict[str, ValueType] = None,
                blocksize: int = 100,
                show_progress_bar: bool = False,
                strip_data: bool = False) -> Tuple[int, int]:

    progressbar = no_progress_bar
    length = None

    if show_progress_bar:
        # Try to import the one from click.
        # People using ase.db will most likely have flask installed
        # and therfore also click.
        try:
            from click import progressbar
        except ImportError:
            pass
        else:
            length = source.count(query_parameters.query)

    nkvp = 0
    nrows = 0
    with destination as db2:
        row_iter = block(source, query_parameters, blocksize)
        with progressbar(row_iter, length=length) as rows:
            for row in rows:
                kvp = row.get('key_value_pairs', {})
                nkvp -= len(kvp)
                kvp.update(add_key_value_pairs)
                nkvp += len(kvp)
                if strip_data:
                    db2.write(row.toatoms(), **kvp)
                else:
                    db2.write(row, data=row.get('data'), **kvp)
                nrows += 1

    return nkvp, nrows
