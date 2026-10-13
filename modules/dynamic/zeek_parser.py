# modules/dynamic/zeek_parser.py
"""
Parser for Zeek `conn.log` files — the interface between network capture and
MalTwin's dynamic (behavioural) analysis.

Why Zeek conn.log?
  - IoT-23 (Stratosphere Lab) ships its labelled IIoT malware traffic as
    `conn.log.labeled` files in this format, so we can train on it directly.
  - The Digital Twin (Module 1, Evaluation 4) will run Zeek on its virtual
    network bridge and produce exactly the same file. The dynamic detector
    therefore does not need to change when the twin comes online.

Format notes
  - Header lines start with '#'. The '#fields' line names the columns.
  - IoT-23 labelled files have a known quirk: the last three columns
    (tunnel_parents, label, detailed-label) are separated by spaces instead
    of tabs, so a data row has 21 tab-separated fields rather than 23.
    We detect and repair that here.
  - Missing values are written as '-' (and '(empty)' for empty sets).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

# Standard Zeek conn.log columns (+ IoT-23 label columns)
ZEEK_CONN_FIELDS = [
    'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto',
    'service', 'duration', 'orig_bytes', 'resp_bytes', 'conn_state',
    'local_orig', 'local_resp', 'missed_bytes', 'history', 'orig_pkts',
    'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'tunnel_parents',
]
LABEL_FIELDS = ['label', 'detailed-label']

NUMERIC_FIELDS = [
    'ts', 'id.orig_p', 'id.resp_p', 'duration', 'orig_bytes', 'resp_bytes',
    'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
]

# Behaviour classes used by the dynamic detector.
# Ordered rules: the first substring that matches the IoT-23 detailed-label wins.
BEHAVIOUR_CLASSES = ['Benign', 'PortScan', 'C&C', 'DDoS', 'Attack', 'FileDownload', 'Okiru']
_DETAILED_LABEL_RULES = [
    ('ddos',              'DDoS'),
    ('filedownload',      'FileDownload'),
    ('partofahorizontal', 'PortScan'),
    ('c&c',               'C&C'),
    ('okiru',             'Okiru'),
    ('attack',            'Attack'),
]


def map_behaviour(label: str, detailed_label: str) -> str:
    """
    Collapse IoT-23's label + detailed-label pair into one behaviour class.

    Examples:
        ('benign', '-')                         → 'Benign'
        ('Malicious', 'PartOfAHorizontalPortScan') → 'PortScan'
        ('Malicious', 'C&C-HeartBeat')          → 'C&C'
        ('Malicious', 'C&C-FileDownload')       → 'FileDownload'
        ('Malicious', 'Okiru-Attack')           → 'Okiru'
        ('Malicious', '-')                      → 'Attack'   (malicious, no detail)
    """
    lab = (label or '').strip().lower()
    det = (detailed_label or '').strip().lower()
    if lab == 'benign':
        return 'Benign'
    for needle, cls in _DETAILED_LABEL_RULES:
        if needle in det:
            # 'Okiru-Attack' should be Okiru, not Attack — handled by rule order
            # only if okiru is checked before attack, which it is.
            return cls
    return 'Attack' if lab == 'malicious' else 'Benign'


def _read_fields_header(lines: Iterable[str]) -> Optional[list[str]]:
    for line in lines:
        if line.startswith('#fields'):
            return line.rstrip('\n').split('\t')[1:]
        if not line.startswith('#'):
            break
    return None


def _split_row(line: str, n_expected: int) -> list[str]:
    parts = line.rstrip('\n').split('\t')
    if len(parts) < n_expected:
        # IoT-23 quirk: trailing fields separated by runs of spaces
        tail = parts[-1].split()
        parts = parts[:-1] + tail
    if len(parts) < n_expected:
        parts += ['-'] * (n_expected - len(parts))
    return parts[:n_expected]


def parse_conn_log(
    source: Union[str, Path, io.TextIOBase, bytes],
    max_rows: Optional[int] = None,
    sample_seed: int = 42,
) -> pd.DataFrame:
    """
    Parse a Zeek conn.log (labelled or not) into a DataFrame.

    Args:
        source:      path, raw bytes (e.g. from a Streamlit upload) or text stream
        max_rows:    if set, keep a uniform random sample of at most this many
                     rows (reservoir sampling — memory-safe on multi-GB files)
        sample_seed: RNG seed for the sample

    Returns:
        DataFrame with Zeek columns, numeric columns coerced to float
        ('-' → NaN), and — if the file is labelled — 'label',
        'detailed-label' and a derived 'behaviour' column.

    Raises:
        ValueError if the input has no recognisable conn.log data.
    """
    if isinstance(source, bytes):
        stream = io.StringIO(source.decode('utf-8', errors='replace'))
    elif isinstance(source, (str, Path)):
        stream = open(source, 'r', encoding='utf-8', errors='replace')
    else:
        stream = source

    try:
        header_lines, first_data = [], None
        for line in stream:
            if line.startswith('#'):
                header_lines.append(line)
                continue
            if line.strip():
                first_data = line
                break
        fields = _read_fields_header(header_lines)
        if fields is None:
            # Allow header-less files if column count matches the standard layout
            fields = ZEEK_CONN_FIELDS + LABEL_FIELDS
        # Normalise IoT-23's merged label header names
        fixed = []
        for f in fields:
            fixed.extend(f.split())
        fields = fixed
        n = len(fields)

        rng = np.random.default_rng(sample_seed)
        rows: list[list[str]] = []
        seen = 0

        def _consume(line: str):
            nonlocal seen
            if not line.strip() or line.startswith('#'):
                return
            row = _split_row(line, n)
            seen += 1
            if max_rows is None or len(rows) < max_rows:
                rows.append(row)
            else:
                j = rng.integers(0, seen)
                if j < max_rows:
                    rows[j] = row

        if first_data is not None:
            _consume(first_data)
        for line in stream:
            _consume(line)
    finally:
        if isinstance(source, (str, Path)):
            stream.close()

    if not rows:
        raise ValueError("No connection records found — is this a Zeek conn.log file?")

    df = pd.DataFrame(rows, columns=fields)
    for col in NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace({'-': np.nan, '(empty)': np.nan}),
                                    errors='coerce')
    if 'label' in df.columns and 'detailed-label' in df.columns:
        df['behaviour'] = [map_behaviour(a, b) for a, b in
                           zip(df['label'], df['detailed-label'])]
    return df


def load_conn_logs(
    paths: Iterable[Union[str, Path]],
    max_rows_per_file: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Parse several conn.log files and concatenate, tagging each row with its capture."""
    frames = []
    for i, p in enumerate(paths):
        df = parse_conn_log(p, max_rows=max_rows_per_file, sample_seed=seed + i)
        df['capture'] = Path(p).parent.name or Path(p).name
        frames.append(df)
    if not frames:
        raise ValueError("No conn.log files given")
    return pd.concat(frames, ignore_index=True)


def write_conn_log(df: pd.DataFrame, path: Union[str, Path], labelled: bool = True) -> None:
    """Write a DataFrame back out in Zeek conn.log format (used by the synthetic generator)."""
    cols = ZEEK_CONN_FIELDS + (LABEL_FIELDS if labelled else [])
    with open(path, 'w', encoding='utf-8') as f:
        f.write('#separator \\x09\n#path\tconn\n')
        f.write('#fields\t' + '\t'.join(cols) + '\n')
        for _, r in df.iterrows():
            vals = []
            for c in cols:
                v = r.get(c, '-')
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    v = '-'
                elif isinstance(v, float) and c not in ('ts', 'duration') and v.is_integer():
                    v = int(v)
                vals.append(str(v))
            f.write('\t'.join(vals) + '\n')
