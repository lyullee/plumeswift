"""Text layouts used by the native QES wind comparison commands."""


_SOLVER_LABELS = {
    'cpu-sor': 'CPU SOR',
    'cuda-sor': 'CUDA SOR',
    'cuda-dynamic-legacy': 'Dynamic',
    'cuda-device-parallel': 'Dynamic (device)',
    'cuda-global-memory': 'Global',
    'cuda-shared-memory': 'Shared',
}


def _table(rows):
    widths = [max(len(cell) for cell in column) for column in zip(*rows, strict=True)]
    ruler = '+'+'+'.join('-'*width for width in widths)+'+'
    lines = [ruler]
    for row in rows:
        lines.append('|'+'|'.join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))+'|')
        lines.append(ruler)
    return '\n'.join(lines)+'\n'


def format_solver_tables(candidates):
    """Render the three native tables for one comparison record."""
    maxima = [('SOLVER NAME', 'MAX U DIFF', 'MAX V DIFF', 'MAX W DIFF')]
    means = [('SOLVER NAME', 'AVG U DIFF', 'AVG V DIFF', 'AVG W DIFF')]
    magnitude = [('SOLVER NAME', 'MAX WVM DIFF', 'AVG WVM DIFF', 'R^2 VALUE')]
    for backend, candidate in candidates.items():
        label = _SOLVER_LABELS[backend]
        fields = candidate['fields']
        speed = candidate['speed']
        maxima.append((label, *(f"{fields[key]['max_abs']:.6f}" for key in 'uvw')))
        means.append((label, *(f"{fields[key]['mean_abs']:.6f}" for key in 'uvw')))
        r_squared = speed['pearson_r_squared']
        magnitude.append((label, f"{speed['max_abs']:.6f}", f"{speed['mean_abs']:.6f}",
                          'nan' if r_squared is None else f'{r_squared:.6f}'))
    tables = ''.join(f'Table {index} of 3\n{_table(rows)}\n'
                     for index, rows in enumerate((maxima, means, magnitude), 1))
    return 'Performing comparative analysis against CPU serial solver...\n\n'+tables+'Comparative analysis complete!\n'


def format_solver_series(result):
    records = result['records']
    if len(records) == 1:
        return format_solver_tables(records[0]['candidates'])
    return '\n'.join(f"Running time step: {record['timestamp']}\n\n"+format_solver_tables(record['candidates'])
                     for record in records)


def format_workspace_stats(result):
    """Render the native field diagnostics in record order."""
    fields = [('terrain', result['terrain'])]
    for record in result['records']:
        fields.extend(record['fields'].items())
    return ''.join(f"stat on {name}\n- total diff = {row['total_abs']:.6g}\n"
                   f"- max diff   = {row['max_abs']:.6g}\n- avg diff   = {row['mean_abs']:.6g}\n\n"
                   for name, row in fields)
