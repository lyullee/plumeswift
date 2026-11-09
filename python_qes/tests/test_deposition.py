from threading import Barrier, Lock, get_ident

from netCDF4 import Dataset
import numpy as np
import pytest

from plumeswift.domain import Domain
from plumeswift.io import write_netcdf
from plumeswift.particles import ParticleIntegrator
from plumeswift.plume_run import run_plume


@pytest.mark.parametrize('parallel', [False, True])
def test_deposition_activation_and_mass_balance(tmp_path, monkeypatch, parallel):
    domain = Domain((12, 12, 10), (1., 1., 1.))
    ws = domain.workspace()
    flags = ws.arrays['icellflag']
    flags[1:4, 4:7, :] = 20
    wind = {name: np.full(domain.face_shape, speed) for name, speed in zip('uvw', (1.25, 0., 0.))}
    wind.update(icellflag=flags, dz_array=ws.arrays['dz_array'])
    wind_file, turb_file = tmp_path/'wind.nc', tmp_path/'turb.nc'
    write_netcdf(wind_file, wind, ws.coordinates)
    stress = dict(txx=.01, tyy=.01, tzz=.01, txy=0., txz=-.002, tyz=0., nuT=.03, CoEps=1e-6,
                  div_tau_x=0., div_tau_y=0., div_tau_z=0.)
    write_netcdf(turb_file, {key: np.full(domain.cell_shape, value) for key, value in stress.items()}, ws.coordinates)
    settings = tmp_path/'plume.xml'
    count = 3 if parallel else 1
    sources = []
    for name, flagged, y, z, mass in [('canopy', 'true', 5.5, 2.5, 10.), ('ground', 'true', 8.5, .5, 10.),
                                      ('cutoff', 'true', 5.5, 2.5, 1e-8), ('control', 'false', 5.5, 2.5, 10.)]:
        sources.append(f'''<baseParticle><tag>{name}</tag><depositionFlag>{flagged}</depositionFlag>
          <source><particleDiameter>100</particleDiameter><particleDensity>1000</particleDensity>
          <releaseType_instantaneous><numPar>{count}</numPar><totalMass>{mass}</totalMass></releaseType_instantaneous>
          <sourceGeometry_Point><posX>5.5</posX><posY>{y}</posY><posZ>{z}</posZ></sourceGeometry_Point>
          </source></baseParticle>''')
    settings.write_text('''<QESPlumeParameters>
      <plumeParameters><timeStep>.25</timeStep><simDur>1</simDur><CourantNumber>.25</CourantNumber>
        <invarianceTol>1e-8</invarianceTol><interpolationMethod>nearestCell</interpolationMethod></plumeParameters>
      <collectionParameters><timeAvgStart>0</timeAvgStart><timeAvgFreq>.5</timeAvgFreq>
        <boxBoundsX1>0</boxBoundsX1><boxBoundsX2>12</boxBoundsX2><nBoxesX>1</nBoxesX>
        <boxBoundsY1>0</boxBoundsY1><boxBoundsY2>12</boxBoundsY2><nBoxesY>1</nBoxesY>
        <boxBoundsZ1>0</boxBoundsZ1><boxBoundsZ2>10</boxBoundsZ2><nBoxesZ>1</nBoxesZ></collectionParameters>
      <boundaryConditions><xBCtype>exiting</xBCtype><yBCtype>exiting</yBCtype><zBCtype>reflection</zBCtype>
        <doDepositions>true</doDepositions></boundaryConditions>
      <particleParameters>'''+''.join(sources)+'''</particleParameters></QESPlumeParameters>''')
    options = dict(turbulence_path=turb_file, native_threads=count if parallel else None)
    baseline = run_plume(wind_file, settings, tmp_path/'off', **options)
    enabled = run_plume(wind_file, settings, tmp_path/'on', deposition=True, **options)
    if parallel:
        workers, lock, barrier = set(), Lock(), Barrier(count, timeout=10)
        integrate = ParticleIntegrator._integrate

        def observe(self, batch, dt, rng, **kwargs):
            if kwargs.get('record') is not None:
                worker = get_ident()
                with lock:
                    first = worker not in workers
                    workers.add(worker)
                if first:
                    barrier.wait()
            return integrate(self, batch, dt, rng, **kwargs)

        monkeypatch.setattr(ParticleIntegrator, '_integrate', observe)
        concurrent = run_plume(wind_file, settings, tmp_path/'parallel', deposition=True,
                               native_parallel=True, **options)
        assert len(workers) == count
        assert concurrent['models'] == enabled['models']
        for tag in ('canopy', 'ground', 'cutoff'):
            filename = f'{tag}_depositionOut.nc'
            with Dataset(tmp_path/'on'/filename) as sequential, Dataset(tmp_path/'parallel'/filename) as actual:
                np.testing.assert_array_equal(sequential['depcvol'][:], actual['depcvol'][:])
    assert not list((tmp_path/'off').glob('*_depositionOut.nc'))
    assert not (tmp_path/'on/control_depositionOut.nc').exists()
    assert enabled['models']['control'] == baseline['models']['control']
    assert baseline['models']['cutoff']['active_particles'] == count
    for tag, y_index in [('canopy', 5), ('ground', 8), ('cutoff', 5)]:
        model = enabled['models'][tag]
        assert model['deposited_mass_kg'] > 0
        assert model['escaped_particles'] == model['rogue_particles'] == 0
        assert model['mass_balance_error_kg'] == pytest.approx(0., abs=1e-16)
        with Dataset(tmp_path/f'on/{tag}_depositionOut.nc') as nc:
            assert nc['depcvol'].units == 'g'
            field = np.asarray(nc['depcvol'][0])
            assert np.all(np.where(field > 0)[1] == y_index)
            assert field.sum()/1000 == pytest.approx(model['deposited_mass_kg'])
    assert enabled['models']['cutoff']['deposition_cutoff_particles'] == count
    assert enabled['models']['cutoff']['active_particles'] == 0
    assert enabled['models']['cutoff']['deposition_cutoff_mass_kg'] > 0
