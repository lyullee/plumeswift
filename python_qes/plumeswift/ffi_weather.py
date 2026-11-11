"""Read DNV's FFI workbooks and average wind as horizontal vectors."""

from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

import numpy as np


_CHANNELS = ('Time', 'Wind_Speed_High', 'Wind_Direction_High', 'Wind_Speed_Low', 'Wind_Direction_Low')


def _raw_wind(path):
    columns = None
    with ZipFile(path) as book:
        sheets = ET.fromstring(book.read('xl/workbook.xml'))
        raw = next((s for s in sheets.findall('.//{*}sheet') if s.get('name', '').casefold() == 'raw data'), None)
        if raw is None:
            raise ValueError(f'{path}: no Raw Data worksheet')
        sheet_id = next((value for key, value in raw.attrib.items() if key.endswith('}id')), None)
        relations = ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))
        target = next((r.get('Target') for r in relations if r.get('Id') == sheet_id), None)
        if target is None:
            raise ValueError(f'{path}: cannot locate the Raw Data worksheet')
        member = target.lstrip('/') if target.startswith('/') else 'xl/' + target
        strings = []
        if 'xl/sharedStrings.xml' in book.namelist():
            strings = [''.join(item.itertext()) for item in ET.fromstring(book.read('xl/sharedStrings.xml'))]
        with book.open(member) as stream:
            parent = None
            for event, element in ET.iterparse(stream, events=('start', 'end')):
                tag = element.tag.rsplit('}', 1)[-1]
                if event == 'start' and tag == 'sheetData':
                    parent = element
                if event != 'end' or tag != 'row':
                    continue
                if columns is None:
                    headings = {}
                    for cell in element:
                        if cell.get('t') == 's' and (value := cell.findtext('{*}v')) is not None:
                            headings[strings[int(value)].strip().casefold()] = cell.get('r').rstrip('0123456789')
                        elif cell.get('t') == 'inlineStr':
                            heading = ''.join(cell.find('{*}is').itertext()).strip().casefold()
                            headings[heading] = cell.get('r').rstrip('0123456789')
                    if all(channel.casefold() in headings for channel in _CHANNELS):
                        columns = {headings[channel.casefold()]: i for i, channel in enumerate(_CHANNELS)}
                else:
                    row = [np.nan] * len(_CHANNELS)
                    for cell in element:
                        column = cell.get('r').rstrip('0123456789')
                        if column in columns and cell.get('t') in (None, 'n'):
                            value = cell.findtext('{*}v')
                            if value is not None:
                                row[columns[column]] = float(value)
                    if np.isfinite(row[0]):
                        yield row
                parent.remove(element)
    if columns is None:
        raise ValueError(f'{path}: raw worksheet is missing FFI wind channels')


def read_weather(path, start, end, interval, reference_height=10):
    """Return time-weighted wind frames [start, end, speed, FROM bearing, ...].

    Mast samples are held until the next timestamp. For each height, records
    with invalid wind channels are excluded. The selected mast height must
    cover at least 90 percent of each frame. Unavailable secondary records
    are marked as missing rather than substituted.
    """
    if not np.isfinite([start, end, interval]).all() or not start < end or interval < 2:
        raise ValueError('FFI weather requires a valid time window and an interval of at least 2 s')
    if reference_height not in (5, 10):
        raise ValueError('The FFI wind mast provides measurements at 5 and 10 m')
    try:
        records = []
        for row in _raw_wind(Path(path)):
            records.append(row)
            if row[0] >= end:
                break
        rows = np.asarray(records, dtype=float).reshape(-1, len(_CHANNELS))
    except (BadZipFile, KeyError, ET.ParseError) as error:
        raise ValueError(f'{path}: cannot read the FFI raw wind worksheet') from error
    if len(rows) < 2:
        raise ValueError(f'{path}: no usable timestamps')
    time = rows[:, 0]
    if np.any(np.diff(time) <= 0) or time[0] > start or time[-1] < end:
        raise ValueError(f'{path}: timestamps must increase and cover {start:g}–{end:g} s')
    # Merge a short tail to avoid judging a sub-sample-duration frame.
    edges = list(np.arange(start, end, interval)) + [end]
    if len(edges) > 2 and edges[-1] - edges[-2] < 2:
        edges.pop(-2)
    frames = []
    for a, b in zip(edges, edges[1:]):
        overlap = np.maximum(0., np.minimum(time[1:], b) - np.maximum(time[:-1], a))
        use = overlap > 0
        if np.any(np.diff(time)[use] > 2):
            raise ValueError(f'{path}: weather data gap in {a:g}–{b:g} s')
        frame = [float(a), float(b)]
        for height, (speed, bearing) in zip((10, 5), (rows[:-1, 1:3].T, rows[:-1, 3:5].T), strict=True):
            valid = use & np.isfinite(speed) & np.isfinite(bearing) & (speed >= 0) & (bearing >= 0) & (bearing <= 360)
            weights = overlap[valid]
            if weights.sum() < .9 * (b - a):
                if height == reference_height:
                    raise ValueError(f'{path}: insufficient {height} m wind coverage in {a:g}–{b:g} s')
                frame.extend((None, None))
                continue
            angle = np.deg2rad(bearing[valid])
            east, north = (np.average(speed[valid] * part, weights=weights) for part in (np.sin(angle), np.cos(angle)))
            frame.extend((float(np.hypot(east, north)), float(np.rad2deg(np.arctan2(east, north)) % 360)))
        frames.append(frame)
    return frames
