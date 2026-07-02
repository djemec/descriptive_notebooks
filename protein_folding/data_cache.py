from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import requests
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa

from protein_tokenizer import ESM_PAD_ID, RES_PAD_ID, ProteinTokenizer


RCSB_SEARCH_URL = 'https://search.rcsb.org/rcsbsearch/v2/query'
RCSB_DOWNLOAD_URL = 'https://files.rcsb.org/download/{pdb_id}.cif'

HEAVY_ATOM_SLOTS = (
    'N',
    'CA',
    'C',
    'O',
    'CB',
    'CG',
    'CG1',
    'CG2',
    'OG',
    'OG1',
    'SG',
    'CD',
    'CD1',
    'CD2',
    'ND1',
    'ND2',
    'OD1',
    'OD2',
    'SD',
    'CE',
    'CE1',
    'CE2',
    'CE3',
    'NE',
    'NE1',
    'NE2',
    'OE1',
    'OE2',
    'CH2',
    'NH1',
    'NH2',
    'OH',
    'CZ',
    'CZ2',
    'CZ3',
    'NZ',
    'OXT',
)
ATOM_SLOT_TO_INDEX = {
    atom_name: atom_index for atom_index, atom_name in enumerate(HEAVY_ATOM_SLOTS)
}
REQUIRED_CHAIN_ATOMS = ('CA',)
MSE_ATOM_RENAMES = {'SE': 'SD'}
ATOM_ELEMENT_VOCAB = {'': 0, 'C': 6, 'N': 7, 'O': 8, 'S': 16}
ATOM_NAME_CHAR_WIDTH = 4

RESIDUE_ATOM_NAMES = {
    'A': ('N', 'CA', 'C', 'O', 'CB'),
    'R': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'),
    'N': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'),
    'D': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'),
    'C': ('N', 'CA', 'C', 'O', 'CB', 'SG'),
    'Q': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'),
    'E': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'),
    'G': ('N', 'CA', 'C', 'O'),
    'H': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'),
    'I': ('N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'),
    'L': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'),
    'K': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'),
    'M': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'),
    'F': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'),
    'P': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'),
    'S': ('N', 'CA', 'C', 'O', 'CB', 'OG'),
    'T': ('N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'),
    'W': (
        'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1',
        'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2',
    ),
    'Y': (
        'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1',
        'CE2', 'CZ', 'OH',
    ),
    'V': ('N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'),
    'X': ('N', 'CA', 'C', 'O', 'CB'),
}

RESIDUE_BONDS = {
    'A': (('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB')),
    'R': (
        ('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'),
        ('CG', 'CD'), ('CD', 'NE'), ('NE', 'CZ'), ('CZ', 'NH1'), ('CZ', 'NH2'),
    ),
    'N': (
        ('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'),
        ('CG', 'OD1'), ('CG', 'ND2'),
    ),
    'D': (
        ('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'),
        ('CG', 'OD1'), ('CG', 'OD2'),
    ),
    'C': (('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'SG')),
    'Q': (
        ('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'),
        ('CG', 'CD'), ('CD', 'OE1'), ('CD', 'NE2'),
    ),
    'E': (
        ('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'),
        ('CG', 'CD'), ('CD', 'OE1'), ('CD', 'OE2'),
    ),
    'G': (('N', 'CA'), ('CA', 'C'), ('C', 'O')),
    'H': (
        ('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'),
        ('CG', 'ND1'), ('CG', 'CD2'), ('ND1', 'CE1'), ('CD2', 'NE2'),
        ('CE1', 'NE2'),
    ),
    'I': (
        ('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG1'),
        ('CB', 'CG2'), ('CG1', 'CD1'),
    ),
    'L': (
        ('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'),
        ('CG', 'CD1'), ('CG', 'CD2'),
    ),
    'K': (
        ('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'),
        ('CG', 'CD'), ('CD', 'CE'), ('CE', 'NZ'),
    ),
    'M': (
        ('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'),
        ('CG', 'SD'), ('SD', 'CE'),
    ),
    'F': (
        ('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'),
        ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'CE1'), ('CD2', 'CE2'),
        ('CE1', 'CZ'), ('CE2', 'CZ'),
    ),
    'P': (
        ('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'),
        ('CG', 'CD'), ('CD', 'N'),
    ),
    'S': (('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'OG')),
    'T': (
        ('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'OG1'),
        ('CB', 'CG2'),
    ),
    'W': (
        ('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'),
        ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'NE1'), ('NE1', 'CE2'),
        ('CD2', 'CE2'), ('CD2', 'CE3'), ('CE2', 'CZ2'), ('CE3', 'CZ3'),
        ('CZ2', 'CH2'), ('CZ3', 'CH2'),
    ),
    'Y': (
        ('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'),
        ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'CE1'), ('CD2', 'CE2'),
        ('CE1', 'CZ'), ('CE2', 'CZ'), ('CZ', 'OH'),
    ),
    'V': (
        ('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG1'),
        ('CB', 'CG2'),
    ),
    'X': (('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB')),
}


def _atom_element(atom_name):
    if atom_name.startswith(('C', 'N', 'O', 'S')):
        return atom_name[0]
    return ''


def _atom_name_chars(atom_name):
    values = np.zeros(ATOM_NAME_CHAR_WIDTH, dtype=np.int32)
    padded = atom_name.ljust(ATOM_NAME_CHAR_WIDTH)[:ATOM_NAME_CHAR_WIDTH]
    for index, value in enumerate(padded):
        values[index] = 0 if value == ' ' else ord(value) - 32
    return values


def _slot_reference_coord(atom_name):
    if atom_name == 'N':
        return np.asarray([-1.25, 0.0, 0.0], dtype=np.float32)
    if atom_name == 'CA':
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    if atom_name == 'C':
        return np.asarray([1.52, 0.0, 0.0], dtype=np.float32)
    if atom_name == 'O':
        return np.asarray([2.10, 0.85, 0.0], dtype=np.float32)
    slot_index = ATOM_SLOT_TO_INDEX[atom_name]
    branch = float((slot_index % 3) - 1)
    depth = 1.2 + 0.55 * float(slot_index // 3)
    return np.asarray([0.15 * branch, depth, 0.75 * branch], dtype=np.float32)


ATOM_ELEMENT_IDS = np.asarray(
    [ATOM_ELEMENT_VOCAB[_atom_element(atom_name)] for atom_name in HEAVY_ATOM_SLOTS],
    dtype=np.int32,
)
ATOM_NAME_CHAR_IDS = np.stack(
    [_atom_name_chars(atom_name) for atom_name in HEAVY_ATOM_SLOTS],
    axis=0,
)
ATOM_REFERENCE_COORDS = np.stack(
    [_slot_reference_coord(atom_name) for atom_name in HEAVY_ATOM_SLOTS],
    axis=0,
)


def _residue_template_features(sequence):
    length = len(sequence)
    atom_count = len(HEAVY_ATOM_SLOTS)
    reference_coords = np.zeros((length, atom_count, 3), dtype=np.float32)
    reference_mask = np.zeros((length, atom_count), dtype=np.bool_)
    atom_to_token = np.full((length, atom_count), -1, dtype=np.int32)
    atom_element = np.zeros((length, atom_count), dtype=np.int32)
    atom_charge = np.zeros((length, atom_count), dtype=np.int32)
    atom_name_chars = np.zeros(
        (length, atom_count, ATOM_NAME_CHAR_WIDTH),
        dtype=np.int32,
    )
    residue_atom_bonds = np.zeros((length, atom_count, atom_count), dtype=np.bool_)
    peptide_bond_mask = np.zeros(length, dtype=np.bool_)

    for residue_index, residue_name in enumerate(sequence):
        expected_atoms = RESIDUE_ATOM_NAMES.get(residue_name, RESIDUE_ATOM_NAMES['X'])
        for atom_name in expected_atoms:
            atom_index = ATOM_SLOT_TO_INDEX[atom_name]
            reference_coords[residue_index, atom_index] = ATOM_REFERENCE_COORDS[atom_index]
            reference_mask[residue_index, atom_index] = True
            atom_to_token[residue_index, atom_index] = residue_index
            atom_element[residue_index, atom_index] = ATOM_ELEMENT_IDS[atom_index]
            atom_name_chars[residue_index, atom_index] = ATOM_NAME_CHAR_IDS[atom_index]
        for atom_a, atom_b in RESIDUE_BONDS.get(residue_name, RESIDUE_BONDS['X']):
            if atom_a not in ATOM_SLOT_TO_INDEX or atom_b not in ATOM_SLOT_TO_INDEX:
                continue
            index_a = ATOM_SLOT_TO_INDEX[atom_a]
            index_b = ATOM_SLOT_TO_INDEX[atom_b]
            residue_atom_bonds[residue_index, index_a, index_b] = True
            residue_atom_bonds[residue_index, index_b, index_a] = True
        if residue_index + 1 < length:
            peptide_bond_mask[residue_index] = True

    return {
        'reference_atom_coords': reference_coords,
        'reference_atom_mask': reference_mask,
        'atom_to_token': atom_to_token,
        'atom_element': atom_element,
        'atom_charge': atom_charge,
        'atom_name_chars': atom_name_chars,
        'residue_atom_bonds': residue_atom_bonds,
        'peptide_bond_mask': peptide_bond_mask,
    }


def build_rcsb_query(min_chain_length=32, max_chain_length=256, max_resolution=3.0, rows=2000):
    return {
        'query': {
            'type': 'group',
            'logical_operator': 'and',
            'nodes': [
                {
                    'type': 'terminal',
                    'service': 'text',
                    'parameters': {
                        'attribute': 'entity_poly.rcsb_entity_polymer_type',
                        'operator': 'exact_match',
                        'value': 'Protein',
                    },
                },
                {
                    'type': 'terminal',
                    'service': 'text',
                    'parameters': {
                        'attribute': 'entity_poly.rcsb_sample_sequence_length',
                        'operator': 'greater_or_equal',
                        'value': int(min_chain_length),
                    },
                },
                {
                    'type': 'terminal',
                    'service': 'text',
                    'parameters': {
                        'attribute': 'entity_poly.rcsb_sample_sequence_length',
                        'operator': 'less_or_equal',
                        'value': int(max_chain_length),
                    },
                },
                {
                    'type': 'terminal',
                    'service': 'text',
                    'parameters': {
                        'attribute': 'exptl.method',
                        'operator': 'exact_match',
                        'value': 'X-RAY DIFFRACTION',
                    },
                },
                {
                    'type': 'terminal',
                    'service': 'text',
                    'parameters': {
                        'attribute': 'rcsb_entry_info.resolution_combined',
                        'operator': 'less_or_equal',
                        'value': float(max_resolution),
                    },
                },
            ],
        },
        'return_type': 'entry',
        'request_options': {
            'paginate': {'start': 0, 'rows': int(rows)},
            'sort': [
                {
                    'sort_by': 'rcsb_accession_info.initial_release_date',
                    'direction': 'asc',
                }
            ],
            'results_content_type': ['experimental'],
        },
    }


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def run_or_load_rcsb_search(
    query,
    query_path,
    response_path,
    refresh=False,
    timeout=60.0,
    allow_request=True,
    session=None,
):
    query_path = Path(query_path)
    response_path = Path(response_path)

    if response_path.exists() and response_path.stat().st_size > 0 and not refresh:
        if not query_path.exists():
            raise ValueError(
                'cached RCSB response has no matching query; refresh the search'
            )
        cached_query = json.loads(query_path.read_text(encoding='utf-8'))
        if cached_query != query:
            raise ValueError(
                'cached RCSB response was produced by a different query; '
                'set refresh=True'
            )
        return json.loads(response_path.read_text(encoding='utf-8'))

    if not allow_request:
        raise FileNotFoundError(
            'no reusable cached RCSB response and network requests are disabled'
        )

    client = session or requests.Session()
    response = client.post(RCSB_SEARCH_URL, json=query, timeout=timeout)
    response.raise_for_status()
    search_response = response.json()
    # Save the response first. If a refresh is interrupted between these writes,
    # the next run sees a query mismatch and refuses unsafe cache reuse.
    save_json(response_path, search_response)
    save_json(query_path, query)
    return search_response


def extract_pdb_ids(search_response):
    identifiers = {
        str(item['identifier']).upper()
        for item in search_response.get('result_set', [])
        if item.get('identifier')
    }
    return sorted(identifiers)


def write_pdb_id_list(path, pdb_ids):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [str(pdb_id).upper() for pdb_id in pdb_ids]
    path.write_text('\n'.join(normalized) + '\n', encoding='ascii')


def download_mmcif(
    pdb_id,
    destination,
    timeout=30.0,
    retries=3,
    backoff_seconds=1.0,
    allow_download=True,
    session=None,
):
    pdb_id = pdb_id.upper()
    destination = Path(destination)
    if destination.suffix.lower() != '.cif':
        destination = destination / f'{pdb_id}.cif'
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        with destination.open('rb') as handle:
            header = handle.read(256).lstrip()
        if header.startswith(b'data_'):
            return {
                'pdb_id': pdb_id,
                'status': 'cached',
                'path': str(destination),
                'bytes': destination.stat().st_size,
            }

    if not allow_download:
        return {
            'pdb_id': pdb_id,
            'status': 'failed',
            'path': str(destination),
            'error': 'no valid cached mmCIF and downloads are disabled',
        }

    client = session or requests.Session()
    temporary = destination.with_suffix(destination.suffix + '.part')
    last_error = None
    for attempt in range(retries):
        try:
            response = client.get(
                RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id),
                timeout=timeout,
            )
            response.raise_for_status()
            content = response.content
            if (
                not content.lstrip().startswith(b'data_')
                or b'_atom_site.' not in content
            ):
                raise ValueError('downloaded response is not an atom-site mmCIF')
            temporary.write_bytes(content)
            if temporary.stat().st_size == 0:
                raise ValueError('downloaded file is empty')
            temporary.replace(destination)
            return {
                'pdb_id': pdb_id,
                'status': 'downloaded',
                'path': str(destination),
                'bytes': destination.stat().st_size,
            }
        except (requests.RequestException, OSError, ValueError) as error:
            last_error = error
            if temporary.exists():
                temporary.unlink()
            if attempt + 1 < retries:
                time.sleep(backoff_seconds * (2**attempt))

    return {
        'pdb_id': pdb_id,
        'status': 'failed',
        'path': str(destination),
        'error': repr(last_error),
    }


def _rejection(pdb_id, chain_id, reason, detail=None):
    result = {
        'pdb_id': pdb_id,
        'chain_id': chain_id,
        'reason': reason,
    }
    if detail:
        result['detail'] = detail
    return result


def _mmcif_column(mmcif_dict, key):
    values = mmcif_dict.get(key, [])
    if isinstance(values, list):
        return [str(value) for value in values]
    return [str(values)]


def _protein_author_chain_ids(mmcif_dict):
    entity_types = _mmcif_column(mmcif_dict, '_entity_poly.type')
    strand_groups = _mmcif_column(mmcif_dict, '_entity_poly.pdbx_strand_id')
    if len(entity_types) != len(strand_groups):
        return set()

    chain_ids = set()
    for entity_type, strand_group in zip(entity_types, strand_groups):
        if not entity_type.lower().startswith('polypeptide'):
            continue
        chain_ids.update(
            chain_id.strip()
            for chain_id in strand_group.split(',')
            if chain_id.strip() not in {'', '.', '?'}
        )
    return chain_ids


def _unobserved_polymer_chain_ids(mmcif_dict, model_number):
    polymer_flags = _mmcif_column(
        mmcif_dict,
        '_pdbx_unobs_or_zero_occ_residues.polymer_flag',
    )
    if not polymer_flags:
        return set()

    model_numbers = _mmcif_column(
        mmcif_dict,
        '_pdbx_unobs_or_zero_occ_residues.PDB_model_num',
    )
    author_chains = _mmcif_column(
        mmcif_dict,
        '_pdbx_unobs_or_zero_occ_residues.auth_asym_id',
    )
    label_chains = _mmcif_column(
        mmcif_dict,
        '_pdbx_unobs_or_zero_occ_residues.label_asym_id',
    )
    row_count = len(polymer_flags)
    if not model_numbers:
        model_numbers = [str(model_number)] * row_count
    if not author_chains:
        author_chains = [''] * row_count
    if not label_chains:
        label_chains = [''] * row_count
    if not (
        len(model_numbers)
        == len(author_chains)
        == len(label_chains)
        == row_count
    ):
        return set()

    chain_ids = set()
    for polymer_flag, raw_model_number, author_chain_id, label_chain_id in zip(
        polymer_flags,
        model_numbers,
        author_chains,
        label_chains,
    ):
        try:
            row_model_number = int(raw_model_number)
        except ValueError:
            continue
        chain_id = (
            author_chain_id
            if author_chain_id not in {'', '.', '?'}
            else label_chain_id
        )
        if (
            polymer_flag.upper() == 'Y'
            and row_model_number == model_number
            and chain_id not in {'', '.', '?'}
        ):
            chain_ids.add(chain_id.strip())
    return chain_ids


def _atom_slot_name(residue_name, atom_name):
    atom_name = atom_name.strip().upper()
    if residue_name == 'MSE':
        atom_name = MSE_ATOM_RENAMES.get(atom_name, atom_name)
    return atom_name


def _is_hydrogen(atom):
    element = str(getattr(atom, 'element', '') or '').strip().upper()
    atom_name = atom.get_name().strip().upper()
    return element == 'H' or atom_name.startswith('H')


def _extract_residue_atoms(residue, one_letter, residue_name):
    atom_count = len(HEAVY_ATOM_SLOTS)
    atom_coords = np.zeros((atom_count, 3), dtype=np.float32)
    atom_mask = np.zeros(atom_count, dtype=np.bool_)
    atom_occupancy = np.zeros(atom_count, dtype=np.float32)
    atom_altloc = np.zeros(atom_count, dtype=np.int32)
    skipped_atoms = []
    zero_occupancy_required = None

    for atom in residue:
        if _is_hydrogen(atom):
            continue
        slot_name = _atom_slot_name(residue_name, atom.get_name())
        slot_index = ATOM_SLOT_TO_INDEX.get(slot_name)
        if slot_index is None:
            skipped_atoms.append(slot_name)
            continue
        occupancy = atom.get_occupancy()
        if occupancy is not None and occupancy <= 0.0:
            if slot_name in REQUIRED_CHAIN_ATOMS:
                zero_occupancy_required = slot_name
                break
            continue
        occupancy_value = float(occupancy) if occupancy is not None else 1.0
        if atom_mask[slot_index] and occupancy_value < atom_occupancy[slot_index]:
            continue
        atom_coords[slot_index] = np.asarray(atom.coord, dtype=np.float32)
        atom_mask[slot_index] = True
        atom_occupancy[slot_index] = occupancy_value
        altloc = str(atom.get_altloc() or '').strip()
        atom_altloc[slot_index] = ord(altloc[:1]) if altloc else 0

    if zero_occupancy_required is not None:
        return None, None, None, None, skipped_atoms, f'{residue_name}: zero occupancy {zero_occupancy_required}'
    for atom_name in REQUIRED_CHAIN_ATOMS:
        slot_index = ATOM_SLOT_TO_INDEX[atom_name]
        if not atom_mask[slot_index]:
            return None, None, None, None, skipped_atoms, f'{residue_name}: missing {atom_name}'
    if not np.isfinite(atom_coords[atom_mask]).all():
        return None, None, None, None, skipped_atoms, 'nonfinite_coordinate'
    return atom_coords, atom_mask, atom_occupancy, atom_altloc, skipped_atoms, None


def _ca_coords_from_atoms(atom_coords, atom_mask):
    ca_index = ATOM_SLOT_TO_INDEX['CA']
    if not atom_mask[:, ca_index].all():
        raise ValueError('all accepted residues must have CA atoms')
    return atom_coords[:, ca_index, :]


def extract_protein_chains(
    cif_path,
    min_chain_length=32,
    max_chain_length=256,
    max_ca_step=5.0,
):
    cif_path = Path(cif_path)
    pdb_id = cif_path.stem.upper()
    # auth_residues=False uses contiguous polymer label indices and excludes
    # ligands and water from the parsed chains.
    label_residue_parser = MMCIFParser(auth_chains=True, auth_residues=False, QUIET=True)
    tokenizer = ProteinTokenizer()

    try:
        structure = label_residue_parser.get_structure(pdb_id, str(cif_path))
        model = next(structure.get_models())
        mmcif_dict = label_residue_parser._mmcif_dict
    except Exception as error:
        return [], [_rejection(pdb_id, '', 'parse_error', repr(error))]

    protein_chain_ids = _protein_author_chain_ids(mmcif_dict)
    model_number = int(getattr(model, 'serial_num', 1))
    incomplete_chain_ids = _unobserved_polymer_chain_ids(mmcif_dict, model_number)

    accepted = []
    rejected = []
    for chain in model:
        chain_id = str(chain.id).strip() or '_'
        declared_protein = chain_id in protein_chain_ids
        residues = []
        residue_atom_coords = []
        residue_atom_masks = []
        residue_atom_occupancies = []
        residue_atom_altlocs = []
        label_seq_ids = []
        unsupported_name = None
        missing_required_atom = None
        skipped_atom_names = set()

        for residue in chain:
            residue_name = residue.get_resname().strip().upper()
            one_letter = tokenizer.pdb_residue_to_one_letter(residue_name)
            if one_letter is None and (declared_protein or is_aa(residue, standard=False)):
                unsupported_name = residue_name
                break
            if one_letter is None:
                continue

            (
                atom_coords,
                atom_mask,
                atom_occupancy,
                atom_altloc,
                skipped_atoms,
                atom_error,
            ) = _extract_residue_atoms(residue, one_letter, residue_name)
            skipped_atom_names.update(skipped_atoms)
            if atom_error is not None:
                missing_required_atom = atom_error
                break
            residues.append(one_letter)
            residue_atom_coords.append(atom_coords)
            residue_atom_masks.append(atom_mask)
            residue_atom_occupancies.append(atom_occupancy)
            residue_atom_altlocs.append(atom_altloc)
            label_seq_ids.append(int(residue.id[1]))

        if unsupported_name is not None:
            rejected.append(
                _rejection(
                    pdb_id,
                    chain_id,
                    'unsupported_modified_residue',
                    unsupported_name,
                )
            )
            continue
        if missing_required_atom is not None:
            rejected.append(
                _rejection(
                    pdb_id,
                    chain_id,
                    'missing_required_atom',
                    missing_required_atom,
                )
            )
            continue
        if not residues:
            rejected.append(_rejection(pdb_id, chain_id, 'not_protein'))
            continue
        if chain_id in incomplete_chain_ids:
            rejected.append(
                _rejection(
                    pdb_id,
                    chain_id,
                    'missing_required_atom',
                    'mmCIF reports an unobserved or zero-occupancy polymer residue',
                )
            )
            continue
        if len(label_seq_ids) > 1:
            label_steps = np.diff(np.asarray(label_seq_ids, dtype=np.int64))
            if not np.all(label_steps == 1):
                rejected.append(
                    _rejection(
                        pdb_id,
                        chain_id,
                        'missing_required_atom',
                        'non-contiguous polymer residue numbering',
                    )
                )
                continue

        length = len(residues)
        if length < min_chain_length:
            rejected.append(_rejection(pdb_id, chain_id, 'too_short', str(length)))
            continue
        if length > max_chain_length:
            rejected.append(_rejection(pdb_id, chain_id, 'too_long', str(length)))
            continue

        atom_coords = np.asarray(residue_atom_coords, dtype=np.float32)
        atom_mask = np.asarray(residue_atom_masks, dtype=np.bool_)
        atom_occupancy = np.asarray(residue_atom_occupancies, dtype=np.float32)
        atom_altloc = np.asarray(residue_atom_altlocs, dtype=np.int32)
        if not np.isfinite(atom_coords[atom_mask]).all():
            rejected.append(
                _rejection(pdb_id, chain_id, 'nonfinite_coordinate')
            )
            continue
        ca_coords = _ca_coords_from_atoms(atom_coords, atom_mask)

        if length > 1:
            steps = ca_coords[1:] - ca_coords[:-1]
            step_distances = np.sqrt((steps * steps).sum(axis=-1))
            largest_step = float(step_distances.max())
            if largest_step > max_ca_step:
                rejected.append(
                    _rejection(
                        pdb_id,
                        chain_id,
                        'chain_break',
                        f'{largest_step:.3f}',
                    )
                )
                continue
        else:
            step_distances = np.zeros(0, dtype=np.float32)

        sequence = ''.join(residues)
        input_ids, res_type = tokenizer.encode(sequence)
        if tokenizer.decode(input_ids) != sequence:
            raise AssertionError('tokenizer round trip failed')
        template_features = _residue_template_features(sequence)
        accepted.append(
            {
                'pdb_id': pdb_id,
                'chain_id': chain_id,
                'sequence': sequence,
                'input_ids': np.asarray(input_ids, dtype=np.int32),
                'res_type': np.asarray(res_type, dtype=np.int32),
                'atom_coords': atom_coords,
                'atom_mask': atom_mask,
                'atom_occupancy': atom_occupancy,
                'atom_altloc': atom_altloc,
                'label_seq_ids': np.asarray(label_seq_ids, dtype=np.int32),
                'skipped_atom_names': sorted(skipped_atom_names),
                'step_distances': step_distances.astype(np.float32),
                **template_features,
            }
        )

    return accepted, rejected


def collect_chain_records(
    cif_paths,
    target_accepted_chains=None,
    min_chain_length=32,
    max_chain_length=256,
    max_ca_step=5.0,
):
    accepted = []
    rejected = []
    for cif_path in sorted(Path(path) for path in cif_paths):
        current_accepted, current_rejected = extract_protein_chains(
            cif_path,
            min_chain_length=min_chain_length,
            max_chain_length=max_chain_length,
            max_ca_step=max_ca_step,
        )
        accepted.extend(current_accepted)
        rejected.extend(current_rejected)
        if (
            target_accepted_chains is not None
            and len(accepted) >= target_accepted_chains
        ):
            break
    return accepted, rejected


def deduplicate_records(records):
    ordered = sorted(
        records,
        key=lambda record: (
            str(record['sequence']),
            str(record['pdb_id']),
            str(record['chain_id']),
        ),
    )
    seen = set()
    unique = []
    duplicates = []
    for record in ordered:
        sequence = str(record['sequence'])
        if sequence in seen:
            duplicates.append(
                _rejection(
                    str(record['pdb_id']),
                    str(record['chain_id']),
                    'duplicate_sequence',
                )
            )
            continue
        seen.add(sequence)
        unique.append(record)
    return unique, duplicates


def split_records(records, train_fraction=0.8, val_fraction=0.1, seed=1337):
    shuffled_records = list(records)
    random.Random(seed).shuffle(shuffled_records)

    train_end = round(len(shuffled_records) * train_fraction)
    val_end = round(len(shuffled_records) * (train_fraction + val_fraction))
    return {
        'train': shuffled_records[:train_end],
        'val': shuffled_records[train_end:val_end],
        'test': shuffled_records[val_end:],
    }


def rejection_counts(rejections):
    return dict(Counter(str(item['reason']) for item in rejections))


def write_shards(
    records,
    split,
    output_root,
    examples_per_shard=1024,
    max_chain_length=256,
    overwrite=False,
):
    if split not in {'train', 'val', 'test'}:
        raise ValueError('split must be train, val, or test')
    if examples_per_shard < 1:
        raise ValueError('examples_per_shard must be positive')

    split_dir = Path(output_root) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    pattern = f'protein_folding_{split}_*'
    existing = sorted(split_dir.glob(pattern))
    if existing and not overwrite:
        raise FileExistsError(
            f'{split_dir} already has generated files; set overwrite=True'
        )
    if overwrite:
        for path in existing:
            path.unlink()

    summaries = []
    tokenizer = ProteinTokenizer()
    for shard_index, start in enumerate(range(0, len(records), examples_per_shard)):
        shard_records = records[start : start + examples_per_shard]
        count = len(shard_records)
        input_ids = np.full(
            (count, max_chain_length),
            ESM_PAD_ID,
            dtype=np.int32,
        )
        res_type = np.full(
            (count, max_chain_length),
            RES_PAD_ID,
            dtype=np.int32,
        )
        atom_count = len(HEAVY_ATOM_SLOTS)
        atom_coords = np.zeros(
            (count, max_chain_length, atom_count, 3),
            dtype=np.float32,
        )
        atom_mask = np.zeros(
            (count, max_chain_length, atom_count),
            dtype=np.bool_,
        )
        reference_atom_coords = np.zeros(
            (count, max_chain_length, atom_count, 3),
            dtype=np.float32,
        )
        reference_atom_mask = np.zeros(
            (count, max_chain_length, atom_count),
            dtype=np.bool_,
        )
        atom_to_token = np.full(
            (count, max_chain_length, atom_count),
            -1,
            dtype=np.int32,
        )
        atom_element = np.zeros(
            (count, max_chain_length, atom_count),
            dtype=np.int32,
        )
        atom_charge = np.zeros(
            (count, max_chain_length, atom_count),
            dtype=np.int32,
        )
        atom_name_chars = np.zeros(
            (count, max_chain_length, atom_count, ATOM_NAME_CHAR_WIDTH),
            dtype=np.int32,
        )
        atom_occupancy = np.zeros(
            (count, max_chain_length, atom_count),
            dtype=np.float32,
        )
        atom_altloc = np.zeros(
            (count, max_chain_length, atom_count),
            dtype=np.int32,
        )
        residue_atom_bonds = np.zeros(
            (count, max_chain_length, atom_count, atom_count),
            dtype=np.bool_,
        )
        peptide_bond_mask = np.zeros((count, max_chain_length), dtype=np.bool_)
        token_bonds = np.zeros(
            (count, max_chain_length, max_chain_length),
            dtype=np.bool_,
        )
        ref_space_uid = np.zeros(
            (count, max_chain_length, atom_count),
            dtype=np.int32,
        )
        residue_index = np.zeros((count, max_chain_length), dtype=np.int32)
        token_index = np.zeros((count, max_chain_length), dtype=np.int32)
        asym_id = np.zeros((count, max_chain_length), dtype=np.int32)
        sym_id = np.zeros((count, max_chain_length), dtype=np.int32)
        entity_id = np.zeros((count, max_chain_length), dtype=np.int32)
        mol_type = np.zeros((count, max_chain_length), dtype=np.int32)
        label_seq_ids = np.zeros((count, max_chain_length), dtype=np.int32)
        residue_mask = np.zeros((count, max_chain_length), dtype=np.bool_)
        lengths = np.zeros(count, dtype=np.int32)
        manifest_rows = []

        for row, record in enumerate(shard_records):
            sequence = str(record['sequence'])
            length = len(sequence)
            if length < 1:
                raise ValueError('records must contain at least one residue')
            if length > max_chain_length:
                raise ValueError(f'record length {length} exceeds shard width')
            encoded = np.asarray(record['input_ids'], dtype=np.int32)
            structural = np.asarray(record['res_type'], dtype=np.int32)
            coordinates = np.asarray(record['atom_coords'], dtype=np.float32)
            coordinate_mask = np.asarray(record['atom_mask'], dtype=np.bool_)
            reference_coordinates = np.asarray(
                record['reference_atom_coords'],
                dtype=np.float32,
            )
            reference_mask = np.asarray(record['reference_atom_mask'], dtype=np.bool_)
            token_map = np.asarray(record['atom_to_token'], dtype=np.int32)
            element_ids = np.asarray(record['atom_element'], dtype=np.int32)
            charges = np.asarray(record['atom_charge'], dtype=np.int32)
            name_chars = np.asarray(record['atom_name_chars'], dtype=np.int32)
            occupancies = np.asarray(record['atom_occupancy'], dtype=np.float32)
            altlocs = np.asarray(record['atom_altloc'], dtype=np.int32)
            bond_mask = np.asarray(record['residue_atom_bonds'], dtype=np.bool_)
            peptide_mask = np.asarray(record['peptide_bond_mask'], dtype=np.bool_)
            label_ids = np.asarray(record['label_seq_ids'], dtype=np.int32)
            if not (
                encoded.shape == (length,)
                and structural.shape == (length,)
                and coordinates.shape == (length, atom_count, 3)
                and coordinate_mask.shape == (length, atom_count)
                and reference_coordinates.shape == (length, atom_count, 3)
                and reference_mask.shape == (length, atom_count)
                and token_map.shape == (length, atom_count)
                and element_ids.shape == (length, atom_count)
                and charges.shape == (length, atom_count)
                and name_chars.shape == (length, atom_count, ATOM_NAME_CHAR_WIDTH)
                and occupancies.shape == (length, atom_count)
                and altlocs.shape == (length, atom_count)
                and bond_mask.shape == (length, atom_count, atom_count)
                and peptide_mask.shape == (length,)
                and label_ids.shape == (length,)
            ):
                raise ValueError('record arrays are not aligned')
            if tokenizer.decode(encoded.tolist()) != sequence:
                raise ValueError('record tokenization does not match sequence')
            expected_input_ids, expected_res_type = tokenizer.encode(sequence)
            if not np.array_equal(
                encoded,
                np.asarray(expected_input_ids, dtype=np.int32),
            ):
                raise ValueError('record input_ids do not match the tokenizer')
            if not np.array_equal(
                structural,
                np.asarray(expected_res_type, dtype=np.int32),
            ):
                raise ValueError('record res_type does not match the tokenizer')
            if not coordinate_mask[:, ATOM_SLOT_TO_INDEX['CA']].all():
                raise ValueError('record is missing a C-alpha atom needed for chain geometry')
            if not np.isfinite(coordinates[coordinate_mask]).all():
                raise ValueError('resolved record coordinates must be finite')
            if not np.all(coordinates[~coordinate_mask] == 0):
                raise ValueError('unresolved record atom coordinates must be zero')
            if not np.isfinite(reference_coordinates[reference_mask]).all():
                raise ValueError('reference atom coordinates must be finite')
            if not np.all(reference_coordinates[~reference_mask] == 0):
                raise ValueError('unexpected reference atom coordinates must be zero')
            expected_token_map = np.where(reference_mask, np.arange(length)[:, None], -1)
            if not np.array_equal(token_map, expected_token_map.astype(np.int32)):
                raise ValueError('atom_to_token does not match residue rows')
            if not np.array_equal(bond_mask, bond_mask.transpose(0, 2, 1)):
                raise ValueError('residue atom bonds must be symmetric')
            if length > 0 and peptide_mask[-1]:
                raise ValueError('last residue cannot have a forward peptide bond')

            input_ids[row, :length] = encoded
            res_type[row, :length] = structural
            atom_coords[row, :length] = coordinates
            atom_mask[row, :length] = coordinate_mask
            reference_atom_coords[row, :length] = reference_coordinates
            reference_atom_mask[row, :length] = reference_mask
            atom_to_token[row, :length] = token_map
            atom_element[row, :length] = element_ids
            atom_charge[row, :length] = charges
            atom_name_chars[row, :length] = name_chars
            atom_occupancy[row, :length] = occupancies
            atom_altloc[row, :length] = altlocs
            residue_atom_bonds[row, :length] = bond_mask
            peptide_bond_mask[row, :length] = peptide_mask
            ref_space_uid[row, :length] = np.where(
                reference_mask,
                token_map.astype(np.int32) + 1,
                0,
            )
            residue_index[row, :length] = label_ids
            token_index[row, :length] = np.arange(length, dtype=np.int32)
            mol_type[row, :length] = 0
            label_seq_ids[row, :length] = label_ids
            residue_mask[row, :length] = True
            lengths[row] = length
            manifest_rows.append(
                {
                    'row': row,
                    'pdb_id': str(record['pdb_id']),
                    'chain_id': str(record['chain_id']),
                    'length': length,
                    'sequence': sequence,
                    'skipped_atom_names': list(record.get('skipped_atom_names', [])),
                }
            )

        stem = f'protein_folding_{split}_{shard_index:06d}'
        shard_path = split_dir / f'{stem}.npz'
        manifest_path = split_dir / f'{stem}.jsonl'
        np.savez_compressed(
            shard_path,
            input_ids=input_ids,
            res_type=res_type,
            atom_coords=atom_coords,
            atom_mask=atom_mask,
            reference_atom_coords=reference_atom_coords,
            reference_atom_mask=reference_atom_mask,
            atom_to_token=atom_to_token,
            atom_element=atom_element,
            atom_charge=atom_charge,
            atom_name_chars=atom_name_chars,
            atom_occupancy=atom_occupancy,
            atom_altloc=atom_altloc,
            residue_atom_bonds=residue_atom_bonds,
            peptide_bond_mask=peptide_bond_mask,
            token_bonds=token_bonds,
            ref_space_uid=ref_space_uid,
            residue_index=residue_index,
            token_index=token_index,
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id,
            mol_type=mol_type,
            label_seq_ids=label_seq_ids,
            residue_mask=residue_mask,
            lengths=lengths,
        )
        manifest_path.write_text(
            ''.join(json.dumps(row, sort_keys=True) + '\n' for row in manifest_rows),
            encoding='utf-8',
        )
        summaries.append(
            {
                'split': split,
                'shard': shard_path.name,
                'manifest': manifest_path.name,
                'examples': count,
                'valid_residues': int(lengths.sum()),
                'array_shape': list(input_ids.shape),
                'atom_slots': list(HEAVY_ATOM_SLOTS),
            }
        )
    return summaries


def load_manifest(path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def validate_shard(shard_path, manifest_path=None, max_ca_step=5.0):
    shard_path = Path(shard_path)
    tokenizer = ProteinTokenizer()
    with np.load(shard_path, allow_pickle=False) as shard:
        arrays = {key: shard[key] for key in shard.files}

    required = {
        'input_ids',
        'res_type',
        'atom_coords',
        'atom_mask',
        'reference_atom_coords',
        'reference_atom_mask',
        'atom_to_token',
        'atom_element',
        'atom_charge',
        'atom_name_chars',
        'atom_occupancy',
        'atom_altloc',
        'residue_atom_bonds',
        'peptide_bond_mask',
        'token_bonds',
        'ref_space_uid',
        'residue_index',
        'token_index',
        'asym_id',
        'sym_id',
        'entity_id',
        'mol_type',
        'label_seq_ids',
        'residue_mask',
        'lengths',
    }
    if set(arrays) != required:
        raise ValueError(f'unexpected shard fields: {sorted(arrays)}')
    if arrays['input_ids'].dtype != np.int32:
        raise TypeError('input_ids must be int32')
    if arrays['res_type'].dtype != np.int32:
        raise TypeError('res_type must be int32')
    if arrays['atom_coords'].dtype != np.float32:
        raise TypeError('atom_coords must be float32')
    if arrays['atom_mask'].dtype != np.bool_:
        raise TypeError('atom_mask must be bool')
    if arrays['reference_atom_coords'].dtype != np.float32:
        raise TypeError('reference_atom_coords must be float32')
    if arrays['reference_atom_mask'].dtype != np.bool_:
        raise TypeError('reference_atom_mask must be bool')
    if arrays['atom_to_token'].dtype != np.int32:
        raise TypeError('atom_to_token must be int32')
    if arrays['atom_element'].dtype != np.int32:
        raise TypeError('atom_element must be int32')
    if arrays['atom_charge'].dtype != np.int32:
        raise TypeError('atom_charge must be int32')
    if arrays['atom_name_chars'].dtype != np.int32:
        raise TypeError('atom_name_chars must be int32')
    if arrays['atom_occupancy'].dtype != np.float32:
        raise TypeError('atom_occupancy must be float32')
    if arrays['atom_altloc'].dtype != np.int32:
        raise TypeError('atom_altloc must be int32')
    if arrays['residue_atom_bonds'].dtype != np.bool_:
        raise TypeError('residue_atom_bonds must be bool')
    if arrays['peptide_bond_mask'].dtype != np.bool_:
        raise TypeError('peptide_bond_mask must be bool')
    if arrays['token_bonds'].dtype != np.bool_:
        raise TypeError('token_bonds must be bool')
    for key in ('ref_space_uid', 'residue_index', 'token_index', 'asym_id', 'sym_id', 'entity_id', 'mol_type'):
        if arrays[key].dtype != np.int32:
            raise TypeError(f'{key} must be int32')
    if arrays['label_seq_ids'].dtype != np.int32:
        raise TypeError('label_seq_ids must be int32')
    if arrays['residue_mask'].dtype != np.bool_:
        raise TypeError('residue_mask must be bool')
    if arrays['lengths'].dtype != np.int32:
        raise TypeError('lengths must be int32')

    count, width = arrays['input_ids'].shape
    atom_count = len(HEAVY_ATOM_SLOTS)
    if arrays['res_type'].shape != (count, width):
        raise ValueError('res_type shape mismatch')
    if arrays['atom_coords'].shape != (count, width, atom_count, 3):
        raise ValueError('atom_coords shape mismatch')
    if arrays['atom_mask'].shape != (count, width, atom_count):
        raise ValueError('atom_mask shape mismatch')
    if arrays['reference_atom_coords'].shape != (count, width, atom_count, 3):
        raise ValueError('reference_atom_coords shape mismatch')
    if arrays['reference_atom_mask'].shape != (count, width, atom_count):
        raise ValueError('reference_atom_mask shape mismatch')
    if arrays['atom_to_token'].shape != (count, width, atom_count):
        raise ValueError('atom_to_token shape mismatch')
    if arrays['atom_element'].shape != (count, width, atom_count):
        raise ValueError('atom_element shape mismatch')
    if arrays['atom_charge'].shape != (count, width, atom_count):
        raise ValueError('atom_charge shape mismatch')
    if arrays['atom_name_chars'].shape != (
        count, width, atom_count, ATOM_NAME_CHAR_WIDTH
    ):
        raise ValueError('atom_name_chars shape mismatch')
    if arrays['atom_occupancy'].shape != (count, width, atom_count):
        raise ValueError('atom_occupancy shape mismatch')
    if arrays['atom_altloc'].shape != (count, width, atom_count):
        raise ValueError('atom_altloc shape mismatch')
    if arrays['residue_mask'].shape != (count, width):
        raise ValueError('residue_mask shape mismatch')
    if arrays['residue_atom_bonds'].shape != (count, width, atom_count, atom_count):
        raise ValueError('residue_atom_bonds shape mismatch')
    if arrays['peptide_bond_mask'].shape != (count, width):
        raise ValueError('peptide_bond_mask shape mismatch')
    if arrays['token_bonds'].shape != (count, width, width):
        raise ValueError('token_bonds shape mismatch')
    if arrays['ref_space_uid'].shape != (count, width, atom_count):
        raise ValueError('ref_space_uid shape mismatch')
    for key in ('residue_index', 'token_index', 'asym_id', 'sym_id', 'entity_id', 'mol_type'):
        if arrays[key].shape != (count, width):
            raise ValueError(f'{key} shape mismatch')
    if arrays['label_seq_ids'].shape != (count, width):
        raise ValueError('label_seq_ids shape mismatch')
    if arrays['lengths'].shape != (count,):
        raise ValueError('lengths shape mismatch')

    manifest = load_manifest(manifest_path) if manifest_path else None
    if manifest is not None and len(manifest) != count:
        raise ValueError('manifest row count mismatch')

    for row, raw_length in enumerate(arrays['lengths']):
        length = int(raw_length)
        if not 0 < length <= width:
            raise ValueError('invalid sequence length')
        if not arrays['residue_mask'][row, :length].all():
            raise ValueError('valid residue mask contains false values')
        if arrays['residue_mask'][row, length:].any():
            raise ValueError('residue padding mask contains true values')
        if not np.all(arrays['input_ids'][row, length:] == ESM_PAD_ID):
            raise ValueError('input padding must use ESM_PAD_ID')
        if not np.all(arrays['res_type'][row, length:] == RES_PAD_ID):
            raise ValueError('res_type padding must use RES_PAD_ID')
        if arrays['atom_mask'][row, length:].any():
            raise ValueError('atom padding mask contains true values')
        if not np.all(arrays['atom_coords'][row, length:] == 0):
            raise ValueError('atom coordinate padding must be zero')
        if arrays['reference_atom_mask'][row, length:].any():
            raise ValueError('reference atom padding mask contains true values')
        if not np.all(arrays['reference_atom_coords'][row, length:] == 0):
            raise ValueError('reference coordinate padding must be zero')
        if not np.all(arrays['atom_to_token'][row, length:] == -1):
            raise ValueError('atom_to_token padding must be -1')
        if not np.all(arrays['atom_element'][row, length:] == 0):
            raise ValueError('atom_element padding must be zero')
        if not np.all(arrays['atom_charge'][row, length:] == 0):
            raise ValueError('atom_charge padding must be zero')
        if not np.all(arrays['atom_name_chars'][row, length:] == 0):
            raise ValueError('atom_name_chars padding must be zero')
        if not np.all(arrays['atom_occupancy'][row, length:] == 0):
            raise ValueError('atom_occupancy padding must be zero')
        if not np.all(arrays['atom_altloc'][row, length:] == 0):
            raise ValueError('atom_altloc padding must be zero')
        if arrays['residue_atom_bonds'][row, length:].any():
            raise ValueError('residue_atom_bonds padding must be false')
        if arrays['peptide_bond_mask'][row, length:].any():
            raise ValueError('peptide_bond_mask padding must be false')
        if arrays['token_bonds'][row, length:].any() or arrays['token_bonds'][row, :, length:].any():
            raise ValueError('token_bonds padding must be false')
        if not np.all(arrays['ref_space_uid'][row, length:] == 0):
            raise ValueError('ref_space_uid padding must be zero')
        for key in ('residue_index', 'token_index', 'asym_id', 'sym_id', 'entity_id', 'mol_type'):
            if not np.all(arrays[key][row, length:] == 0):
                raise ValueError(f'{key} padding must be zero')
        if not np.all(arrays['label_seq_ids'][row, length:] == 0):
            raise ValueError('label_seq_ids padding must be zero')

        valid_ids = arrays['input_ids'][row, :length]
        if not np.isin(valid_ids, np.arange(3, 24, dtype=np.int32)).all():
            raise ValueError('valid input IDs must be in the protein range 3-23')
        valid_res_type = arrays['res_type'][row, :length]
        if not np.isin(
            valid_res_type,
            np.arange(2, 23, dtype=np.int32),
        ).all():
            raise ValueError('valid res_type IDs must be in the protein range 2-22')
        valid_atom_coords = arrays['atom_coords'][row, :length]
        valid_atom_mask = arrays['atom_mask'][row, :length]
        reference_coords = arrays['reference_atom_coords'][row, :length]
        reference_mask = arrays['reference_atom_mask'][row, :length]
        atom_to_token = arrays['atom_to_token'][row, :length]
        atom_element = arrays['atom_element'][row, :length]
        atom_charge = arrays['atom_charge'][row, :length]
        atom_name_chars = arrays['atom_name_chars'][row, :length]
        atom_occupancy = arrays['atom_occupancy'][row, :length]
        atom_altloc = arrays['atom_altloc'][row, :length]
        residue_bonds = arrays['residue_atom_bonds'][row, :length]
        peptide_mask = arrays['peptide_bond_mask'][row, :length]
        token_bonds = arrays['token_bonds'][row, :length, :length]
        ref_space_uid = arrays['ref_space_uid'][row, :length]
        residue_index = arrays['residue_index'][row, :length]
        token_index = arrays['token_index'][row, :length]
        asym_id = arrays['asym_id'][row, :length]
        sym_id = arrays['sym_id'][row, :length]
        entity_id = arrays['entity_id'][row, :length]
        mol_type = arrays['mol_type'][row, :length]
        label_ids = arrays['label_seq_ids'][row, :length]
        if not valid_atom_mask[:, ATOM_SLOT_TO_INDEX['CA']].all():
            raise ValueError('every accepted residue must have a C-alpha atom')
        if not np.isfinite(valid_atom_coords[valid_atom_mask]).all():
            raise ValueError('resolved atom coordinates must be finite')
        if not np.all(valid_atom_coords[~valid_atom_mask] == 0):
            raise ValueError('unresolved atom coordinates must be zero')
        if not np.isfinite(reference_coords[reference_mask]).all():
            raise ValueError('reference atom coordinates must be finite')
        if not np.all(reference_coords[~reference_mask] == 0):
            raise ValueError('unexpected reference atom coordinates must be zero')
        expected_token_map = np.where(reference_mask, np.arange(length)[:, None], -1)
        if not np.array_equal(atom_to_token, expected_token_map.astype(np.int32)):
            raise ValueError('atom_to_token does not match reference atom rows')
        if not np.array_equal(atom_element, reference_mask * ATOM_ELEMENT_IDS[None, :]):
            raise ValueError('atom_element does not match atom slots')
        if not np.all(atom_charge == 0):
            raise ValueError('first atom cache expects neutral atom_charge values')
        if not np.array_equal(
            atom_name_chars,
            reference_mask[..., None] * ATOM_NAME_CHAR_IDS[None, :, :],
        ):
            raise ValueError('atom_name_chars does not match atom slots')
        if not np.all((atom_occupancy >= 0.0) & (atom_occupancy <= 1.0)):
            raise ValueError('atom occupancy values must be between zero and one')
        if not np.all(atom_occupancy[~valid_atom_mask] == 0):
            raise ValueError('unresolved atom occupancy must be zero')
        if not np.all(atom_altloc[~valid_atom_mask] == 0):
            raise ValueError('unresolved atom altloc must be zero')
        if not np.array_equal(residue_bonds, residue_bonds.transpose(0, 2, 1)):
            raise ValueError('residue atom bonds must be symmetric')
        if length > 0 and peptide_mask[-1]:
            raise ValueError('last residue cannot have a forward peptide bond')
        if length > 1 and not peptide_mask[: length - 1].all():
            raise ValueError('contiguous protein residues should have peptide bonds')
        if not np.array_equal(token_bonds, token_bonds.T):
            raise ValueError('token_bonds must be symmetric')
        expected_token_bonds = np.zeros((length, length), dtype=np.bool_)
        if not np.array_equal(token_bonds, expected_token_bonds):
            raise ValueError('standard protein token_bonds must be zero')
        expected_ref_space_uid = np.where(
            reference_mask,
            atom_to_token.astype(np.int32) + 1,
            0,
        )
        if not np.array_equal(ref_space_uid, expected_ref_space_uid):
            raise ValueError('ref_space_uid must match atom-to-token reference spaces')
        if not np.all(label_ids > 0):
            raise ValueError('valid label_seq_ids must be positive')
        if not np.array_equal(residue_index, label_ids):
            raise ValueError('residue_index must match label_seq_ids')
        if not np.array_equal(token_index, np.arange(length, dtype=np.int32)):
            raise ValueError('token_index must be contiguous')
        if asym_id.any() or sym_id.any() or entity_id.any() or mol_type.any():
            raise ValueError('single-chain protein cache expects zero chain/entity/mol metadata')
        if length > 1 and not np.all(np.diff(label_ids) == 1):
            raise ValueError('label_seq_ids must be contiguous')
        if length > 1:
            ca_coords = valid_atom_coords[:, ATOM_SLOT_TO_INDEX['CA'], :]
            steps = ca_coords[1:] - ca_coords[:-1]
            step_distances = np.sqrt((steps * steps).sum(axis=-1))
            if float(step_distances.max()) > max_ca_step + 1e-5:
                raise ValueError('cached chain contains a large C-alpha break')

        decoded = tokenizer.decode(valid_ids.tolist())
        expected_input_ids, expected_res_type = tokenizer.encode(decoded)
        if not np.array_equal(
            valid_ids,
            np.asarray(expected_input_ids, dtype=np.int32),
        ):
            raise ValueError('input IDs do not match decoded sequence')
        if not np.array_equal(
            valid_res_type,
            np.asarray(expected_res_type, dtype=np.int32),
        ):
            raise ValueError('res_type does not match decoded sequence')
        if manifest is not None:
            manifest_row = manifest[row]
            if manifest_row.get('row') != row:
                raise ValueError('manifest row index mismatch')
            if manifest_row.get('length') != length:
                raise ValueError('manifest length mismatch')
            if decoded != manifest_row.get('sequence'):
                raise ValueError('decoded sequence does not match manifest')
            if not manifest_row.get('pdb_id') or not manifest_row.get('chain_id'):
                raise ValueError('manifest identifiers must be nonempty')

    return {
        'path': str(shard_path),
        'examples': count,
        'width': width,
        'valid_residues': int(arrays['lengths'].sum()),
        'fields': sorted(arrays),
    }


def dataset_statistics(splits):
    all_records = [record for records in splits.values() for record in records]
    amino_acids = Counter()
    atom_observations = Counter()
    lengths = []
    step_distances = []
    for record in all_records:
        amino_acids.update(str(record['sequence']))
        if 'atom_mask' in record:
            atom_mask = np.asarray(record['atom_mask'], dtype=np.bool_)
            for atom_index, atom_name in enumerate(HEAVY_ATOM_SLOTS):
                atom_observations[atom_name] += int(atom_mask[:, atom_index].sum())
        lengths.append(len(str(record['sequence'])))
        step_distances.extend(
            np.asarray(record['step_distances'], dtype=np.float32).tolist()
        )
    return {
        'split_counts': {split: len(records) for split, records in splits.items()},
        'lengths': lengths,
        'amino_acid_counts': dict(sorted(amino_acids.items())),
        'atom_observation_counts': dict(sorted(atom_observations.items())),
        'step_distances': step_distances,
        'test_full_length_64': sum(
            len(str(record['sequence'])) <= 64 for record in splits['test']
        ),
        'test_crop_64': sum(
            len(str(record['sequence'])) > 64 for record in splits['test']
        ),
    }


def build_dataset_metadata(
    query,
    selected_pdb_ids,
    splits,
    rejections,
    shard_summaries,
    random_seed,
    min_chain_length,
    max_chain_length,
    max_resolution,
    max_ca_step,
):
    return {
        'cache_contract_version': 3,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'RCSB Protein Data Bank',
        'structure_representation': 'fixed heavy-atom slots',
        'atom_slots': list(HEAVY_ATOM_SLOTS),
        'atom_element_vocab': ATOM_ELEMENT_VOCAB,
        'atom_name_char_width': ATOM_NAME_CHAR_WIDTH,
        'required_chain_atoms': list(REQUIRED_CHAIN_ATOMS),
        'bond_representation': 'per-residue atom-slot adjacency plus forward peptide bond mask',
        'query': query,
        'selected_pdb_ids': selected_pdb_ids,
        'selected_pdb_entry_count': len(selected_pdb_ids),
        'random_seed': random_seed,
        'filters': {
            'experimental_method': 'X-RAY DIFFRACTION',
            'maximum_resolution_angstrom': max_resolution,
            'minimum_chain_length': min_chain_length,
            'maximum_chain_length': max_chain_length,
            'maximum_adjacent_ca_distance_angstrom': max_ca_step,
            'modified_residues': 'MSE maps to MET; other modifications rejected',
            'hydrogens': 'ignored',
        },
        'split_rule': {
            'algorithm': 'seeded shuffle of accepted records',
            'train_fraction': 0.8,
            'val_fraction': 0.1,
            'test_fraction': 0.1,
        },
        'split_counts': {split: len(records) for split, records in splits.items()},
        'rejection_counts': rejection_counts(rejections),
        'shards': shard_summaries,
    }
