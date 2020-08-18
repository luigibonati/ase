"""Module to read and write atoms in cif file format.

See http://www.iucr.org/resources/cif/spec/version1.1/cifsyntax for a
description of the file format.  STAR extensions as save frames,
global blocks, nested loops and multi-data values are not supported.
The "latin-1" encoding is required by the IUCR specification.
"""

import io
import re
import shlex
import warnings
from typing import Dict, List, Tuple, Optional, Union, Iterator, Any
import collections.abc

import numpy as np

from ase import Atoms
from ase.cell import Cell
from ase.parallel import paropen
from ase.spacegroup import crystal
from ase.spacegroup.spacegroup import spacegroup_from_data, Spacegroup
from ase.io.cif_unicode import format_unicode, handle_subscripts


rhombohedral_spacegroups = {146, 148, 155, 160, 161, 166, 167}


# Old conventions:
old_spacegroup_names = {'Abm2': 'Aem2',
                        'Aba2': 'Aea2',
                        'Cmca': 'Cmce',
                        'Cmma': 'Cmme',
                        'Ccca': 'Ccc1'}

# CIF maps names to either single values or to multiple values via loops.
CIFDataValue = Union[str, int, float]
CIFData = Union[CIFDataValue, List[CIFDataValue]]


def convert_value(value: str) -> CIFDataValue:
    """Convert CIF value string to corresponding python type."""
    value = value.strip()
    if re.match('(".*")|(\'.*\')$', value):
        return handle_subscripts(value[1:-1])
    elif re.match(r'[+-]?\d+$', value):
        return int(value)
    elif re.match(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$', value):
        return float(value)
    elif re.match(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\(\d+\)$',
                  value):
        return float(value[:value.index('(')])  # strip off uncertainties
    elif re.match(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\(\d+$',
                  value):
        warnings.warn('Badly formed number: "{0}"'.format(value))
        return float(value[:value.index('(')])  # strip off uncertainties
    else:
        return handle_subscripts(value)


def parse_multiline_string(lines: List[str], line: str) -> str:
    """Parse semicolon-enclosed multiline string and return it."""
    assert line[0] == ';'
    strings = [line[1:].lstrip()]
    while True:
        line = lines.pop().strip()
        if line[:1] == ';':
            break
        strings.append(line)
    return '\n'.join(strings).strip()


def parse_singletag(lines: List[str], line: str) -> Tuple[str, CIFDataValue]:
    """Parse a CIF tag (entries starting with underscore). Returns
    a key-value pair."""
    kv = line.split(None, 1)
    if len(kv) == 1:
        key = line
        line = lines.pop().strip()
        while not line or line[0] == '#':
            line = lines.pop().strip()
        if line[0] == ';':
            value = parse_multiline_string(lines, line)
        else:
            value = line
    else:
        key, value = kv
    return key, convert_value(value)


def parse_loop(lines: List[str]) -> Dict[str, List[CIFDataValue]]:
    """Parse a CIF loop. Returns a dict with column tag names as keys
    and a lists of the column content as values."""
    headers: List[str] = []
    line = lines.pop().strip()
    while line.startswith('_'):
        tokens = line.split()
        headers.append(tokens[0].lower())
        if len(tokens) == 1:
            line = lines.pop().strip()
        else:
            line = ' '.join(tokens[1:])
            break
    columns: Dict[str, List[CIFDataValue]] = dict([(header, [])
                                                   for header in headers])
    if len(columns) != len(headers):
        seen = set()
        duplicates = []
        for header in headers:
            if headers in seen:
                duplicates.append(header)
            else:
                seen.add(header)
        warnings.warn('Duplicated loop tags: {0}'.format(duplicates))

    tokens = []
    while True:
        lowerline = line.lower()
        if (not line or
            line.startswith('_') or
            lowerline.startswith('data_') or
            lowerline.startswith('loop_')):
            break
        if line.startswith('#'):
            line = lines.pop().strip()
            continue
        if line.startswith(';'):
            t = [parse_multiline_string(lines, line)]
        else:
            if len(headers) == 1:
                t = [line]
            else:
                t = shlex.split(line, posix=False)

        line = lines.pop().strip()

        tokens.extend(t)
        if len(tokens) < len(columns):
            continue
        if len(tokens) == len(headers):
            for header, token in zip(headers, tokens):
                columns[header].append(convert_value(token))
        else:
            warnings.warn('Wrong number of tokens: {0}'.format(tokens))
        tokens = []
    if line:
        lines.append(line)
    return columns


def parse_items(lines: List[str], line: str) -> Dict[str, CIFData]:
    """Parse a CIF data items and return a dict with all tags."""
    tags: Dict[str, CIFData] = {}
    while True:
        if not lines:
            break
        line = lines.pop()
        if not line:
            break
        line = line.strip()
        lowerline = line.lower()
        if not line or line.startswith('#'):
            continue
        elif line.startswith('_'):
            key, value = parse_singletag(lines, line)
            tags[key.lower()] = value
        elif lowerline.startswith('loop_'):
            tags.update(parse_loop(lines))
        elif lowerline.startswith('data_'):
            if line:
                lines.append(line)
            break
        elif line.startswith(';'):
            parse_multiline_string(lines, line)
        else:
            raise ValueError('Unexpected CIF file entry: "{0}"'.format(line))
    return tags


class CIFBlock(collections.abc.Mapping):
    """A block (i.e., a single system) in a crystallographic information file.

    Use this object to query CIF tags or import information as ASE objects."""

    cell_tags = ['_cell_length_a', '_cell_length_b', '_cell_length_c',
                 '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma']

    def __init__(self, name: str, tags: Dict[str, CIFData]):
        self.name = name
        self._tags = tags

    def __repr__(self) -> str:
        tags = set(self._tags)
        return f'CIFBlock({self.name}, tags={tags})'

    def __getitem__(self, key: str) -> CIFData:
        return self._tags[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tags)

    def __len__(self) -> int:
        return len(self._tags)

    def get(self, key, default=None):
        return self._tags.get(key, default)

    def get_cellpar(self) -> Optional[List]:
        try:
            return [self[tag] for tag in self.cell_tags]
        except KeyError:
            return None

    def get_cell(self) -> Cell:
        cellpar = self.get_cellpar()
        if cellpar is None:
            return Cell.new([0, 0, 0])
        return Cell.new(cellpar)

    def _raw_scaled_positions(self) -> Optional[np.ndarray]:
        coords = [self.get(name) for name in ['_atom_site_fract_x',
                                              '_atom_site_fract_y',
                                              '_atom_site_fract_z']]
        if None in coords:
            return None
        return np.array(coords).T

    def _raw_positions(self) -> Optional[np.ndarray]:
        coords = [self.get('_atom_site_cartn_x'),
                  self.get('_atom_site_cartn_y'),
                  self.get('_atom_site_cartn_z')]
        if None in coords:
            return None
        return np.array(coords).T

    def get_scaled_positions(self):
        scaled_positions = self._raw_scaled_positions()
        if scaled_positions is None:
            positions = self._raw_positions()
            if positions is None:
                raise RuntimeError('No positions found in structure')
            cell = self.get_cell()
            scaled_positions = cell.scaled_positions(positions)
        return scaled_positions

    def _get_symbols_with_deuterium(self):
        labels = self._get_any(['_atom_site_type_symbol',
                                '_atom_site_label'])
        assert labels is not None
        symbols = []
        for label in labels:
            # Strip off additional labeling on chemical symbols
            match = re.search(r'([A-Z][a-z]?)', label)
            symbol = match.group(0)
            symbols.append(symbol)
        return symbols

    def get_symbols(self) -> List[str]:
        symbols = self._get_symbols_with_deuterium()
        return [symbol if symbol != 'D' else 'H' for symbol in symbols]

    def _where_deuterium(self):
        return np.array([symbol == 'D' for symbol
                         in self._get_symbols_with_deuterium()], bool)

    def _get_masses(self) -> Optional[np.ndarray]:
        mask = self._where_deuterium()
        if not any(mask):
            return None

        symbols = self.get_symbols()
        masses = Atoms(symbols).get_masses()
        masses[mask] = 2.01355
        return masses

    def _get_any(self, names):
        for name in names:
            if name in self:
                return self[name]
        return None

    def _get_spacegroup_number(self):
        # Symmetry specification, see
        # http://www.iucr.org/resources/cif/dictionaries/cif_sym for a
        # complete list of official keys.  In addition we also try to
        # support some commonly used depricated notations
        return self._get_any(['_space_group.it_number',
                              '_space_group_it_number',
                              '_symmetry_int_tables_number'])

    def _get_spacegroup_name(self):
        hm_symbol = self._get_any(['_space_group.Patterson_name_h-m',
                                   '_space_group.patterson_name_h-m',
                                   '_symmetry_space_group_name_h-m',
                                   '_space_group_name_h-m_alt'])

        hm_symbol = old_spacegroup_names.get(hm_symbol, hm_symbol)
        return hm_symbol

    def _get_sitesym(self):
        sitesym = self._get_any(['_space_group_symop_operation_xyz',
                                 '_space_group_symop.operation_xyz',
                                 '_symmetry_equiv_pos_as_xyz'])
        if isinstance(sitesym, str):
            sitesym = [sitesym]
        return sitesym

    def _get_fractional_occupancies(self):
        return self.get('_atom_site_occupancy')

    def _get_setting(self) -> Optional[int]:
        setting_str = self.get('_symmetry_space_group_setting')
        if setting_str is None:
            return None

        setting = int(setting_str)
        if setting not in [1, 2]:
            raise ValueError(
                f'Spacegroup setting must be 1 or 2, not {setting}')
        return setting

    def get_spacegroup(self, subtrans_included) -> Spacegroup:
        # XXX The logic in this method needs serious cleaning up!
        # The setting needs to be passed as either 1 or two, not None (default)
        no = self._get_spacegroup_number()
        hm_symbol = self._get_spacegroup_name()
        sitesym = self._get_sitesym()

        setting = 1
        spacegroup = 1
        if sitesym is not None:
            subtrans = [(0.0, 0.0, 0.0)] if subtrans_included else None
            spacegroup = spacegroup_from_data(
                no=no, symbol=hm_symbol, sitesym=sitesym, subtrans=subtrans,
                setting=setting)
        elif no is not None:
            spacegroup = no
        elif hm_symbol is not None:
            spacegroup = hm_symbol
        else:
            spacegroup = 1

        setting_std = self._get_setting()

        setting_name = None
        if '_symmetry_space_group_setting' in self:
            assert setting_std is not None
            setting = setting_std
        elif '_space_group_crystal_system' in self:
            setting_name = self['_space_group_crystal_system']
        elif '_symmetry_cell_setting' in self:
            setting_name = self['_symmetry_cell_setting']

        if setting_name:
            no = Spacegroup(spacegroup).no
            if no in rhombohedral_spacegroups:
                if setting_name == 'hexagonal':
                    setting = 1
                elif setting_name in ('trigonal', 'rhombohedral'):
                    setting = 2
                else:
                    warnings.warn(
                        'unexpected crystal system %r for space group %r' % (
                            setting_name, spacegroup))
            # FIXME - check for more crystal systems...
            else:
                warnings.warn(
                    'crystal system %r is not interpreted for space group %r. '
                    'This may result in wrong setting!' % (
                        setting_name, spacegroup))

        spg = Spacegroup(spacegroup)
        if no is not None:
            assert int(spg) == no, (int(spg), no)
        assert spg.setting == setting, (spg.setting, setting)
        return spg

    def get_unsymmetrized_structure(self) -> Atoms:
        return Atoms(symbols=self.get_symbols(),
                     cell=self.get_cell(),
                     masses=self._get_masses(),
                     scaled_positions=self.get_scaled_positions())

    def get_atoms(self, store_tags=False, primitive_cell=False,
                  subtrans_included=True, fractional_occupancies=True) -> Atoms:
        """Returns an Atoms object from a cif tags dictionary.  See read_cif()
        for a description of the arguments."""
        if primitive_cell and subtrans_included:
            raise RuntimeError(
                'Primitive cell cannot be determined when sublattice '
                'translations are included in the symmetry operations listed '
                'in the CIF file, i.e. when `subtrans_included` is True.')

        cell = self.get_cell()
        assert cell.rank in [0, 3]

        kwargs: Dict[str, Any] = {}
        if store_tags:
            kwargs['info'] = self._tags.copy()

        if fractional_occupancies:
            occupancies = self._get_fractional_occupancies()
        else:
            occupancies = None

        if occupancies is not None:
            # no warnings in this case
            kwargs['onduplicates'] = 'keep'

        # The unsymmetrized_structure is not the asymmetric unit
        # because the asymmetric unit should have (in general) a smaller cell,
        # whereas we have the full cell.
        unsymmetrized_structure = self.get_unsymmetrized_structure()

        if cell.rank == 3:
            spacegroup = self.get_spacegroup(subtrans_included)
            atoms = crystal(unsymmetrized_structure,
                            spacegroup=spacegroup,
                            setting=spacegroup.setting,
                            occupancies=occupancies,
                            primitive_cell=primitive_cell,
                            **kwargs)
        else:
            atoms = unsymmetrized_structure
            if kwargs.get('info') is not None:
                atoms.info.update(kwargs['info'])
            if occupancies is not None:
                # Compile an occupancies dictionary
                occ_dict = {}
                for i, sym in enumerate(atoms.symbols):
                    occ_dict[i] = {sym: occupancies[i]}
                atoms.info['occupancy'] = occ_dict

        return atoms


def parse_block(lines: List[str], line: str) -> CIFBlock:
    assert line.lower().startswith('data_')
    blockname = line.split('_', 1)[1].rstrip()
    tags = parse_items(lines, line)
    return CIFBlock(blockname, tags)


def parse_cif(fileobj, reader='ase') -> Iterator[CIFBlock]:
    if reader == 'ase':
        return parse_cif_ase(fileobj)
    elif reader == 'pycodcif':
        return parse_cif_pycodcif(fileobj)
    else:
        raise ValueError(f'No such reader: {reader}')


def parse_cif_ase(fileobj) -> Iterator[CIFBlock]:
    """Parse a CIF file using ase CIF parser."""

    if isinstance(fileobj, str):
        with open(fileobj, 'rb') as fileobj:
            data = fileobj.read()
    else:
        data = fileobj.read()

    if isinstance(data, bytes):
        data = data.decode('latin1')
    data = format_unicode(data)
    lines = [e for e in data.split('\n') if len(e) > 0]
    if len(lines) > 0 and lines[0].rstrip() == '#\\#CIF_2.0':
        warnings.warn('CIF v2.0 file format detected; `ase` CIF reader might '
                      'incorrectly interpret some syntax constructions, use '
                      '`pycodcif` reader instead')
    lines = [''] + lines[::-1]    # all lines (reversed)

    while lines:
        line = lines.pop().strip()
        if not line or line.startswith('#'):
            continue

        yield parse_block(lines, line)


def parse_cif_pycodcif(fileobj) -> Iterator[CIFBlock]:
    """Parse a CIF file using pycodcif CIF parser."""
    if not isinstance(fileobj, str):
        fileobj = fileobj.name

    try:
        from pycodcif import parse
    except ImportError:
        raise ImportError(
            'parse_cif_pycodcif requires pycodcif ' +
            '(http://wiki.crystallography.net/cod-tools/pycodcif/)')

    data, _, _ = parse(fileobj)

    for datablock in data:
        tags = datablock['values']
        for tag in tags.keys():
            values = [convert_value(x) for x in tags[tag]]
            if len(values) == 1:
                tags[tag] = values[0]
            else:
                tags[tag] = values
        yield CIFBlock(datablock['name'], tags)


def read_cif(fileobj, index, store_tags=False, primitive_cell=False,
             subtrans_included=True, fractional_occupancies=True,
             reader='ase') -> Iterator[Atoms]:
    """Read Atoms object from CIF file. *index* specifies the data
    block number or name (if string) to return.

    If *index* is None or a slice object, a list of atoms objects will
    be returned. In the case of *index* is *None* or *slice(None)*,
    only blocks with valid crystal data will be included.

    If *store_tags* is true, the *info* attribute of the returned
    Atoms object will be populated with all tags in the corresponding
    cif data block.

    If *primitive_cell* is true, the primitive cell will be built instead
    of the conventional cell.

    If *subtrans_included* is true, sublattice translations are
    assumed to be included among the symmetry operations listed in the
    CIF file (seems to be the common behaviour of CIF files).
    Otherwise the sublattice translations are determined from setting
    1 of the extracted space group.  A result of setting this flag to
    true, is that it will not be possible to determine the primitive
    cell.

    If *fractional_occupancies* is true, the resulting atoms object will be
    tagged equipped with an array `occupancy`. Also, in case of mixed
    occupancies, the atom's chemical symbol will be that of the most dominant
    species.

    String *reader* is used to select CIF reader. Value `ase` selects
    built-in CIF reader (default), while `pycodcif` selects CIF reader based
    on `pycodcif` package.
    """
    # Find all CIF blocks with valid crystal data
    images = []
    for block in parse_cif(fileobj, reader):
        atoms = block.get_atoms(
            store_tags, primitive_cell,
            subtrans_included,
            fractional_occupancies=fractional_occupancies)
        images.append(atoms)
    for atoms in images[index]:
        yield atoms


def split_chem_form(comp_name):
    """Returns e.g. AB2  as ['A', '1', 'B', '2']"""
    split_form = re.findall(r'[A-Z][a-z]*|\d+',
                            re.sub(r'[A-Z][a-z]*(?![\da-z])',
                                   r'\g<0>1', comp_name))
    return split_form


def format_cell(cell: Cell) -> str:
    assert cell.rank == 3
    lines = []
    for name, value in zip(CIFBlock.cell_tags, cell.cellpar()):
        line = '{:20} {:g}\n'.format(name, value)
        lines.append(line)
    assert len(lines) == 6
    return ''.join(lines)


def format_generic_spacegroup_info() -> str:
    # We assume no symmetry whatsoever
    return '\n'.join([
        '_symmetry_space_group_name_H-M    "P 1"',
        '_symmetry_int_tables_number       1',
        '',
        'loop_',
        '  _symmetry_equiv_pos_as_xyz',
        "  'x, y, z'",
        '',
    ])


def write_cif(fd, images, cif_format='default',
              wrap=True, labels=None, loop_keys=None) -> None:
    """Write *images* to CIF file.

    wrap: bool
        Wrap atoms into unit cell.

    labels: list
        Use this list (shaped list[i_frame][i_atom] = string) for the
        '_atom_site_label' section instead of automatically generating
        it from the element symbol.

    loop_keys: dict
        Add the information from this dictionary to the `loop_`
        section.  Keys are printed to the `loop_` section preceeded by
        ' _'. dict[key] should contain the data printed for each atom,
        so it needs to have the setup `dict[key][i_frame][i_atom] =
        string`. The strings are printed as they are, so take care of
        formating. Information can be re-read using the `store_tags`
        option of the cif reader.

    """

    if loop_keys is None:
        loop_keys = {}

    if isinstance(fd, str):
        fd = paropen(fd, 'wb')

    fd = io.TextIOWrapper(fd, encoding='latin-1')

    if hasattr(images, 'get_positions'):
        images = [images]

    for i_frame, atoms in enumerate(images):
        fd.write('data_image%d\n' % i_frame)

        if cif_format == 'mp':
            comp_name = atoms.get_chemical_formula(mode='reduce')
            sf = split_chem_form(comp_name)
            formula_sum = ''
            ii = 0
            while ii < len(sf):
                formula_sum = formula_sum + ' ' + sf[ii] + sf[ii + 1]
                ii = ii + 2

            formula_sum = str(formula_sum)
            fd.write('_chemical_formula_structural       %s\n' %
                     atoms.get_chemical_formula(mode='reduce'))
            fd.write('_chemical_formula_sum      "%s"\n' %
                     formula_sum)

        # Do this only if there's three non-zero lattice vectors
        if atoms.cell.rank == 3:
            fd.write(format_cell(atoms.cell))
            fd.write('\n')
            fd.write(format_generic_spacegroup_info())
            fd.write('\n')

        fd.write('loop_\n')

        # Is it a periodic system?
        coord_type = 'fract' if atoms.pbc.all() else 'Cartn'

        if cif_format == 'mp':
            fd.write('  _atom_site_type_symbol\n')
            fd.write('  _atom_site_label\n')
            fd.write('  _atom_site_symmetry_multiplicity\n')
            fd.write('  _atom_site_{0}_x\n'.format(coord_type))
            fd.write('  _atom_site_{0}_y\n'.format(coord_type))
            fd.write('  _atom_site_{0}_z\n'.format(coord_type))
            fd.write('  _atom_site_occupancy\n')
        else:
            fd.write('  _atom_site_label\n')
            fd.write('  _atom_site_occupancy\n')
            fd.write('  _atom_site_{0}_x\n'.format(coord_type))
            fd.write('  _atom_site_{0}_y\n'.format(coord_type))
            fd.write('  _atom_site_{0}_z\n'.format(coord_type))
            fd.write('  _atom_site_thermal_displace_type\n')
            fd.write('  _atom_site_B_iso_or_equiv\n')
            fd.write('  _atom_site_type_symbol\n')

        if coord_type == 'fract':
            coords = atoms.get_scaled_positions(wrap).tolist()
        else:
            coords = atoms.get_positions(wrap).tolist()
        symbols = atoms.get_chemical_symbols()
        occupancies = [1 for i in range(len(symbols))]

        # try to fetch occupancies // spacegroup_kinds - occupancy mapping
        try:
            occ_info = atoms.info['occupancy']
            kinds = atoms.arrays['spacegroup_kinds']
        except KeyError:
            pass
        else:
            for i, kind in enumerate(kinds):
                occupancies[i] = occ_info[kind][symbols[i]]
                # extend the positions array in case of mixed occupancy
                for sym, occ in occ_info[kind].items():
                    if sym != symbols[i]:
                        symbols.append(sym)
                        coords.append(coords[i])
                        occupancies.append(occ)

        # Can only do it now since length of atoms is not always equal to the
        # number of entries.
        # Do not move this up!
        extra_data = ["" for i in range(len(symbols))]
        for key in loop_keys:
            extra_data = ["{}  {}".format(
                extra_data[i], loop_keys[key][i_frame][i])
                for i in range(len(symbols))]
            fd.write("  _{}\n".format(key))

        if labels:
            included_labels = labels[i_frame]
        else:
            no: Dict[str, int] = {}
            included_labels = []
            for symbol in symbols:
                if symbol in no:
                    no[symbol] += 1
                else:
                    no[symbol] = 1
                included_labels.append('%s%d' % (symbol, no[symbol]))

        assert len(symbols) == len(coords) == len(
            occupancies) == len(included_labels) == len(extra_data)

        for symbol, pos, occ, label, ext in zip(
                symbols, coords, occupancies, included_labels, extra_data):
            if cif_format == 'mp':
                fd.write('  %-2s  %4s  %4s  %7.5f  %7.5f  %7.5f  %6.1f%s\n' %
                         (symbol, label, 1,
                          pos[0], pos[1], pos[2], occ, ext))
            else:
                fd.write(
                    '  %-8s %6.4f %7.5f  %7.5f  %7.5f  %4s  %6.3f  %-2s%s\n'
                    % (label, occ, pos[0], pos[1], pos[2],
                       'Biso', 1.0, symbol, ext))

    # Using the TextIOWrapper somehow causes the file to close
    # when this function returns.
    # Detach in order to circumvent this highly illogical problem:
    fd.detach()
