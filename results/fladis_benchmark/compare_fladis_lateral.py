"""Compare the two-dimensional mast-direction run against the x-only mast run."""
import json
import math

import numpy as np

from compare_fladis_mast import ROOT, TRIALS, read, plot, write_csv, cross_arc_centroid


OUT = ROOT/'lateral_wind'
AFTER = OUT/'pps120'
BEFORE = ROOT/'mast_wind'


def main():
    old_arcs = {(int(r['trial']),int(r['arc_m'])):r for r in read(BEFORE/'arc_comparison.csv')}
    old_points = read(BEFORE/'sensor_comparison.csv')
    arcs, points, stations = [], [], []
    for trial, cfg in TRIALS.items():
        run = json.loads((AFTER/f'fladis{trial}_run.json').read_text())
        budget = run['accounting']
        assert run['particles_per_s']==120 and run['dt_s']==.25
        assert abs(budget['mass_balance_error_kg'])<1e-6
        assert math.isclose(budget['released_mass_kg'],cfg['rate']*cfg['window'],abs_tol=1e-7)
        assert budget['lateral_slope_gradient_per_m']>0
        old = [r for r in old_points if int(r['trial'])==trial]
        new = read(AFTER/f'fladis{trial}_sensors.csv')
        for a,b in zip(old,new,strict=True):
            for old_key,new_key in (('arc_m','nominal_arc_m'),('x_m','x_m'),('y_m','y_m'),
                                    ('z_m','z_m'),('observed_ppmv','observed_ppmv')):
                assert float(a[old_key])==float(b[new_key])
            points.append(dict(trial=trial,arc_m=b['nominal_arc_m'],x_m=b['x_m'],
                               y_m=b['y_m'],z_m=b['z_m'],observed_ppmv=b['observed_ppmv'],
                               before_ppmv=a['after_ppmv'],after_ppmv=b['model_ppmv']))
        for x,y,z,direction,sigma in budget['mast_station_xy_z_fromdeg_sigma']:
            stations.append(dict(trial=trial,x_m=x,y_m=y,z_m=z,from_deg=direction,
                                 sigma_deg=sigma,
                                 slope_gradient_per_m=budget['lateral_slope_gradient_per_m']))
        for b in read(AFTER/f'fladis{trial}_arcs.csv'):
            arc=int(b['nominal_arc_m'])
            a=old_arcs[trial,arc]
            line=[r for r in new if int(r['nominal_arc_m'])==arc and
                  float(r['z_m'])==float(b['height_m'])]
            arcs.append(dict(trial=trial,arc_m=arc,height_m=b['height_m'],sensor_n=b['n'],
                             observed_center_m=a['observed_center_m'],
                             before_center_m=a['after_center_m'],
                             after_center_m=cross_arc_centroid(line,'model_ppmv'),
                             observed_width_m=a['observed_width_m'],
                             before_width_m=a['after_width_m'],after_width_m=b['modeled_arc_width_m'],
                             observed_matched_width_m=b['observed_matched_width_m'],
                             before_matched_width_m=a['after_matched_width_m'],
                             after_matched_width_m=b['modeled_matched_width_m'],
                             observed_peak_ppmv=a['observed_peak_ppmv'],
                             before_peak_ppmv=a['after_peak_ppmv'],after_peak_ppmv=b['modeled_arcmax_ppmv'],
                             flacs_D_peak_ppmv=a['flacs_D_peak_ppmv'],
                             flacs_min_peak_ppmv=a['flacs_min_peak_ppmv'],
                             flacs_max_peak_ppmv=a['flacs_max_peak_ppmv'],
                             observed_matched_peak_ppmv=b['observed_sensor_peak_at_height_ppmv'],
                             before_matched_peak_ppmv=a['after_matched_peak_ppmv'],
                             after_matched_peak_ppmv=b['modeled_sensor_peak_at_height_ppmv'],
                             before_sensor_mae_ppmv=a['after_sensor_mae_ppmv'],after_sensor_mae_ppmv=b['mae']))
    assert len(arcs)==9 and len(points)==153 and len(stations)==12
    write_csv(OUT/'arc_comparison.csv',arcs)
    write_csv(OUT/'sensor_comparison.csv',points)
    write_csv(OUT/'mast_inputs.csv',stations)
    metrics=[]
    for selection in (20,70,240,'all'):
        a=[r for r in arcs if selection=='all' or r['arc_m']==selection]
        p=[r for r in points if selection=='all' or int(r['arc_m'])==selection]
        for stage in ('before','after'):
            metrics.append(dict(scope=str(selection),stage=stage,
                center_mae_m=float(np.mean([abs(float(r[f'{stage}_center_m'])-float(r['observed_center_m'])) for r in a])),
                width_mae_m=float(np.mean([abs(float(r[f'{stage}_width_m'])-float(r['observed_width_m'])) for r in a])),
                sensor_mae_ppmv=float(np.mean([abs(float(r[f'{stage}_ppmv'])-float(r['observed_ppmv'])) for r in p])),
                peak_geometric_error_factor=float(np.exp(np.mean([abs(np.log(float(r[f'{stage}_peak_ppmv'])/float(r['observed_peak_ppmv']))) for r in a])))))
    write_csv(OUT/'metrics.csv',metrics)
    plot(arcs,output=OUT,labels=('Mast means by x','Mast means by x and y'))


if __name__=='__main__':
    main()
