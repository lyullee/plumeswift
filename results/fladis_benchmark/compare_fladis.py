"""Pair field measurements and the Gexcon submission with model results."""
import csv
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_fladis import INPUT, ROOT, TRIALS, stats, write_csv


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def flacs():
    out = {}
    with ZipFile(INPUT/'flacs.zip') as archive:
        name = 'flacs_gexcon_lMauri/requested_arcmax_conc_meashts.txt'
        for raw in archive.read(name).decode().splitlines():
            line = raw.strip()
            if line.startswith('FLADIS'):
                case, label = line.split(',')
                trial, stability = int(case.removeprefix('FLADIS')), label.split()[-1]
            elif line and line[0].isdigit():
                distance, height, value = map(float, line.split(','))
                out[trial, stability, 240 if distance == 238 else int(distance)] = (distance, height, value)
    return out


def main():
    gexcon = flacs()
    rows, sensors, score = [], [], []
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey=False, layout='constrained')
    for i, trial in enumerate(TRIALS):
        points = read(ROOT/f'fladis{trial}_sensors.csv')
        sensors.extend(points)
        for j, arc in enumerate(read(ROOT/f'fladis{trial}_arcs.csv')):
            nominal = int(arc['nominal_arc_m'])
            distance, height, neutral = gexcon[trial, 'D', nominal]
            assert height == float(arc['height_m'])
            alternatives = [gexcon[trial, stability, nominal][2]
                            for stability in ('C', 'D', 'E')
                            if (trial, stability, nominal) in gexcon]
            rows.append(dict(**arc, flacs_distance_m=distance, flacs_cerc_D_ppmv=neutral,
                             flacs_cerc_min_ppmv=min(alternatives), flacs_cerc_max_ppmv=max(alternatives)))
            data = [r for r in points if int(r['nominal_arc_m']) == nominal and float(r['z_m']) == height]
            data.sort(key=lambda r: float(r['y_m']))
            y = [float(r['y_m']) for r in data]
            ax = axes[i,j]
            ax.plot(y, [float(r['observed_ppmv']) for r in data], 'o-', label='Observed sensors')
            ax.plot(y, [float(r['model_ppmv']) for r in data], 's-', label='PlumeSwift at sensors')
            ax.axhline(neutral, color='0.42', ls='--', label='FLACS arc maximum (D)')
            ax.set_title(f'FLADIS {trial} | {nominal} m arc, z={height:g} m')
            ax.grid(alpha=.25)
        score.append(dict(trial=trial, **stats([float(r['observed_ppmv']) for r in points],
                                               [float(r['model_ppmv']) for r in points])))
    write_csv(ROOT/'arc_comparison.csv', rows)
    write_csv(ROOT/'sensor_comparison.csv', sensors)
    write_csv(ROOT/'trial_metrics.csv', score)
    axes[0,0].legend(fontsize=8)
    fig.supxlabel('Cross-arc coordinate (m)')
    fig.supylabel('Mean NH₃ (ppmv)')
    fig.savefig(ROOT/'fladis_profiles.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
