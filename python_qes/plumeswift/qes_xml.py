"""Read QES inputs with the historical lexical variants shipped upstream."""
from pathlib import Path
from math import isfinite
import re
import xml.etree.ElementTree as ET


_TOKENS = re.compile(
    rb'(?P<comment><!--.*?-->|<!~-.*?-->)'
    rb'|(?P<protected><!\[CDATA\[.*?\]\]>|<\?.*?\?>)'
    rb'|(?P<geometry><sourceGeometry_FullDomain>(?:\s*)</SourceGeometry_FullDomain>)',
    re.DOTALL,
)
_ROOTS = frozenset(('QESWindsParameters', 'QESPlumeParameters', 'sensor', 'fires'))
_STREAM_NUMBERS = {
    int: re.compile(r'\s*([+-]?[0-9]+)', re.ASCII),
    float: re.compile(r'\s*([+-]?(?>[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+|(?![eE])))', re.ASCII),
}
# Midpoint where formatted extraction rounds beyond the finite float32 range.
_FLOAT32_OVERFLOW = float.fromhex('0x1.ffffffp127')


def _repair_token(match):
    token = match[0]
    if match.lastgroup == 'comment' and (token.startswith(b'<!~-') or b'--' in token[4:-3]):
        # Boost discards the entire comment/declaration through the first -->.
        return re.sub(rb'[^\r\n]', b' ', token)
    if match.lastgroup == 'geometry':
        return token.replace(b'</SourceGeometry_FullDomain>', b'</sourceGeometry_FullDomain>')
    return token


def read_qes_xml(path):
    """Parse XML strictly first and repair only known QES lexical variants."""
    data = Path(path).read_bytes()
    try:
        return ET.fromstring(data)
    except ET.ParseError as error:
        if b'<!DOCTYPE' in data:
            raise
        repaired = _TOKENS.sub(_repair_token, data)
        if repaired == data:
            raise
        try:
            root = ET.fromstring(repaired)
        except ET.ParseError:
            raise error from None
        if root.tag not in _ROOTS:
            raise error
        return root


def parse_optional_number(element, name, default, *, convert, legacy=False):
    """Read an optional scalar, using the Boost default on failed conversion."""
    text = element.findtext(name)
    if text is None:
        return default
    if not legacy:
        return convert(text)
    match = _STREAM_NUMBERS[convert].fullmatch(text.rstrip(' \t\r\n\v\f'))
    if match is None:
        return default
    try:
        value = convert(match[1])
    except ValueError:
        return default
    if convert is int:
        return value if -(2**31) <= value < 2**31 else default
    return value if isfinite(value) else default


def parse_vector(element, name, *, convert=float, size=None, repeated=False, stop_on_invalid=False):
    """Read a whitespace-separated QES vector or a repeated-value list."""
    entries = element.findall(name)
    if not repeated:
        entries = entries[:1]
    values = []
    stream_pattern = _STREAM_NUMBERS.get(convert) if stop_on_invalid else None
    if stream_pattern is not None:
        # Formatted extraction accepts a numeric prefix and leaves any suffix
        # in the stream for the next extraction.
        text = ' '.join(child.text or '' for child in entries)
        offset = 0
        while match := stream_pattern.match(text, offset):
            try:
                value = convert(match[1])
            except ValueError:
                break
            if convert is int and not -(2**31) <= value < 2**31:
                break
            if convert is float and abs(value) >= _FLOAT32_OVERFLOW:
                break
            values.append(value)
            offset = match.end()
    else:
        for token in (token for child in entries for token in (child.text or '').split()):
            try:
                values.append(convert(token))
            except ValueError:
                if not stop_on_invalid:
                    raise
                break
    if size is not None and len(values) != size:
        raise ValueError(f'{name} requires exactly {size} values')
    return tuple(values)
