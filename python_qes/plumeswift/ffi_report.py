"""Read the outdoor sensor means from a user-supplied FFI report."""

import re


REPORT = 'FFI-RAPPORT 20/03101'
REPORT_URL = ('https://www.ffi.no/en/publications-archive/'
              'large-scale-leakage-of-liquid-hydrogen-lh2-tests-related-to-bunkering-and-maritime-use-of-liquid-hydrogen')
# Zero-based PDF pages of the Appendix A concentration tables.
TABLE_PAGES = (78, 82, 86, 90, 94, 100, 106)
_NUMBER = r'-?\d+(?:[.,]\d+)?'
_ROW = re.compile(
    rf'^\s*(OC_\d{{2}})\s*\(R\s*=\s*({_NUMBER}),\s*Z\s*=\s*({_NUMBER}),'
    rf'\s*B\s*=\s*({_NUMBER})\)\s+({_NUMBER})(?:\s+{_NUMBER}){{3}}\s*%vol\s*$',
    re.MULTILINE | re.IGNORECASE,
)


def _number(value):
    return float(value.replace(',', '.'))


def parse_sensor_table(text, test):
    if not re.search(rf'Test\s+Name\s+Test0{test}\b', text):
        raise ValueError(f'Expected Appendix A concentration table for FFI Test {test}')
    start = re.search(rf'\bStart\s+({_NUMBER})\s+sec\b', text)
    end = re.search(rf'\bEnd\s+({_NUMBER})\s+sec\b', text)
    rows = sorted(_ROW.findall(text))
    if not start or not end or [row[0] for row in rows] != [f'OC_{i:02}' for i in range(1, 31)]:
        raise ValueError(f'Could not read the averaging window and 30 distinct sensors for FFI Test {test}')
    sensors = [dict(id=sensor, radius_m=_number(radius), bearing_deg=_number(bearing),
                    height_m=_number(height), observed_volpct=_number(average))
               for sensor, radius, height, bearing, average in rows]
    return [_number(start[1]), _number(end[1])], sensors


def read_ffi_report(path, tests):
    """Extract tables in memory. The user obtains the PDF directly from FFI."""
    try:
        from pypdf import PdfReader
        from pypdf.constants import UserAccessPermissions
        from pypdf.errors import PyPdfError
    except ImportError as error:
        raise ValueError('Install PDF support with: python -m pip install "plumeswift[ffi]"') from error

    try:
        reader = PdfReader(path)
        permissions = reader.user_access_permissions
        if permissions is not None and not permissions & UserAccessPermissions.EXTRACT:
            raise ValueError('This PDF does not permit text extraction')
        cover = ' '.join((reader.pages[0].extract_text() + reader.pages[1].extract_text()).split())
        if '20/03101' not in cover or len(reader.pages) <= TABLE_PAGES[-1]:
            raise ValueError(f'Use the complete official {REPORT} PDF')
        return {test: parse_sensor_table(reader.pages[TABLE_PAGES[test - 1]].extract_text(), test)
                for test in tests}
    except (PyPdfError, IndexError) as error:
        raise ValueError(f'Could not read the official {REPORT} PDF: {error}') from error
