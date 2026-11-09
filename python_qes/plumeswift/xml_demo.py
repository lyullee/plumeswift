"""The two standalone QES scratch XML parser examples."""
from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


class MissingValue(ValueError):
    pass


class RepeatedValueError(ValueError):
    pass


# Boost's default parser pairs end tags by nesting depth. The simulation
# fixture contains a mismatched closing name, and Boost accepts that file.
_XML_TAGS = re.compile(
    rb'<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?.*?\?>'
    rb'|</(?P<end>[^\s<>/]+)\s*>'
    rb'|<(?P<start>[^\s<>/!?]+)(?:[^<>"\']|"[^"]*"|\'[^\']*\')*>',
    re.DOTALL,
)
_DECLARATION = re.compile(rb'^(?:\xef\xbb\xbf)?\s*<\?xml\s+.*?\?>', re.DOTALL)
_INT = re.compile(r'[+-]?[0-9]+')
_FLOAT = re.compile(r'[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?')
_SPACE = re.compile(r'[ \t\r\n\v\f]*')


def _document(path):
    data = Path(path).read_bytes()
    try:
        document = ET.Element('root')
        document.append(ET.fromstring(data))
        return document
    except ET.ParseError:
        # Keep malformed XML handling local to these historical examples.
        stack = []

        def end_name(match):
            if match['start'] and not match[0].endswith(b'/>'):
                stack.append(match['start'])
            if match['end'] and stack:
                return b'</'+stack.pop()+b'>'
            return match[0]

        body = _DECLARATION.sub(b'', data)
        document = ET.fromstring(b'<root>'+_XML_TAGS.sub(end_name, body)+b'</root>')
        if _text(document).strip():
            raise ET.ParseError('Text outside XML elements')
        return document


def _text(element):
    return (element.text or '')+''.join(child.tail or '' for child in element)


def _convert(text, kind):
    if kind == 'string':
        return text
    text = text.strip()
    if kind == 'char' and len(text.encode('utf-8')) == 1:
        return text
    if kind == 'int' and _INT.fullmatch(text):
        value = int(text)
        if -(1 << 31) <= value < (1 << 31):
            return value
    if kind == 'float' and _FLOAT.fullmatch(text):
        with np.errstate(over='ignore', invalid='ignore'):
            value = np.float32(text)
        if np.isfinite(value):
            return float(value)
    raise ValueError(f'Invalid {kind} value')


@dataclass(frozen=True)
class _Node:
    element: ET.Element
    path: str = 'root'

    def missing(self, tag):
        return MissingValue(f'Could not find {self.path}::{tag}')

    def child(self, tag, *, required=True):
        node = self.element.find(tag)
        if node is None:
            if required:
                raise self.missing(tag)
            return None
        return _Node(node, f'{self.path}::{tag}')

    def children(self, tag):
        return [_Node(node, f'{self.path}::{tag}') for node in self.element.findall(tag)]

    def value(self, tag, kind, default=None):
        node = self.child(tag, required=default is None)
        if node is not None:
            try:
                return _convert(_text(node.element), kind)
            except ValueError:
                if default is None:
                    raise self.missing(tag) from None
        return default


def _a(node):
    children = node.children('AVal')
    if not children:
        raise node.missing('AVal')
    try:
        return [_convert(_text(child.element), 'int') for child in children]
    except ValueError:
        raise RepeatedValueError(f'Invalid integer in {node.path}::AVal') from None


def _polymorph(node):
    name = node.value('name', 'string')
    if node.element.tag == 'P1':
        return ('P1', name, node.value('xVal', 'int'))
    character = node.value('cVal', 'char')
    value = node.value('fVal', 'float')
    node.value('qVal', 'float')
    return ('P2', name, character, value)


def read_xml_demo(path):
    """Return the output rows in the original example's order."""
    x = _Node(_document(path)).child('X')
    rows = ['X', x.value('intVal', 'int'), 'A', *_a(x.child('A')), 'B']
    b = x.child('B')
    rows.append(b.value('numAs', 'int'))
    repeated = b.children('A')
    if not repeated:
        raise b.missing('A')
    for a in repeated:
        rows.extend(('A', *_a(a)))
    c = x.child('C')
    rows.extend(('C', c.value('intValx', 'int'), c.value('floatValy', 'float')))
    rows.extend(_polymorph(x.child('P1', required=False) or x.child('P2')))
    d = x.child('D')
    choices = d.children('P1')+d.children('P2')
    if not choices:
        raise d.missing('P2')
    rows.append('D')
    for choice in choices:
        rows.extend(_polymorph(choice))
    return rows


def _vector(element, kind):
    """Read numeric prefixes until the C++ stream would fail."""
    text = _text(element)
    pattern = _INT if kind == 'int' else _FLOAT
    values = []
    cursor = 0
    while cursor < len(text):
        token = pattern.match(text, _SPACE.match(text, cursor).end())
        if token is None:
            break
        cursor = token.end()
        if kind == 'float' and text[cursor:cursor+1] in ('e', 'E'):
            break
        try:
            values.append(_convert(token[0], kind))
        except ValueError:
            break
    return values


def read_simulation_demo(path):
    """Return the historical example's parsed parameters, if present."""
    sim = _Node(_document(path)).child('simulationParameters', required=False)
    if sim is None:
        return None
    values = {tag: _vector(sim.child(tag).element, kind)
              for tag, kind in (('domain', 'int'), ('cellSize', 'float'))}
    required = ('verticalStretching', 'totalTimeIncrements', 'UTCConversion', 'Epoch', 'rooftopFlag',
                'upwindCavityFlag', 'streetCanyonFlag', 'streetIntersectionFlag', 'wakeFlag',
                'sidewallFlag', 'maxIterations', 'residualReduction', 'useDiffusion', 'domainRotation')
    for tag in required:
        values[tag] = sim.value(tag, 'float' if tag in ('Epoch', 'domainRotation') else 'int')
    for tag in ('UTMX', 'UTMY', 'UTMZone', 'UTMZoneLetter', 'quicCDFFlag', 'explosiveDamageFlag', 'buildingArrayFlag'):
        values[tag] = sim.value(tag, 'int', default=0)
    return values


def run_xml_demo(path, *, simulation=False):
    """Format the original output and the two handled error classes."""
    try:
        if simulation:
            read_simulation_demo(path)
            return '', '', 0
        rows = read_xml_demo(path)
    except (OSError, ET.ParseError):
        return '', 'Error reading tree\n', 0
    except MissingValue as error:
        return '', f'ERROR: {error}\n', 0
    except RepeatedValueError as error:
        return '', f'ERROR: {error}\n', 1
    return ''.join(f'{value:.6g}\n' if isinstance(value, float) else f'{value}\n' for value in rows), '', 0
