import numpy as np
import pytest
from plumeswift.thermo import LegacyEquilibrium,CoolPropEquilibrium
from plumeswift.contracts import ModelDomainError

def test_legacy_energy_and_ranges():
    model=LegacyEquilibrium(relative_humidity=0)
    c,T,rho,x,q=model.table.T;y=c/rho
    residual=(1-y)*1005*(T-276.45)+y*(14300*(T-20.37)+445000*(q-.441))
    assert np.max(abs(residual))<1e-6
    assert np.all(np.diff(c)>0)
    assert np.all((q>=0)&(q<=1))

def test_coolprop_adiabatic_energy_and_table_bounds():
    from CoolProp.CoolProp import PropsSI
    model=CoolPropEquilibrium(relative_humidity=0)
    c,T,rho,x,q=model.table.T;y=c/rho;p=101325.
    h0=model.metadata['source_specific_enthalpy_J_kg']
    residual=y*(PropsSI('Hmass','T',T,'P',p,'Hydrogen')-h0)+(1-y)*(PropsSI('Hmass','T',T,'P',p,'Air')-PropsSI('Hmass','T',276.45,'P',p,'Air'))
    assert np.max(abs(residual))<1e-5
    assert np.all(q==1)
    with pytest.raises(ModelDomainError):model.evaluate(np.array([model.table[-1,0]*1.01]))

def test_interchangeable_thermo_and_explicit_metadata():
    from plumeswift.assembly import assemble,AssemblyConfig,ComponentChoice
    for name in ['thermo.legacy','thermo.coolprop']:
        config=AssemblyConfig(ComponentChoice(name),ComponentChoice('poisson.krylov'))
        components=assemble(config);state=components.thermo.evaluate(np.array([0,.001,.01,.05]))
        assert np.isfinite(state.density).all()
        assert state.temperature[0]==pytest.approx(276.45)
        assert components.manifest()['configuration']['physics']['lh2_drag_coefficient']==.5

def test_nested_property_mixture_and_moisture_components():
    from plumeswift.assembly import assemble,AssemblyConfig,ComponentChoice
    choice=ComponentChoice('thermo.coolprop',{
        'properties':ComponentChoice('properties.coolprop',{'hydrogen_fluid':'ParaHydrogen'}),
        'mixing_rule':ComponentChoice('mixing.amagat'),
        'moisture':ComponentChoice('moisture.trace')})
    model=assemble(AssemblyConfig(choice,ComponentChoice('poisson.redblack')))
    assert model.thermo.describe()['property_component']['hydrogen_fluid']=='ParaHydrogen'
    assert model.thermo.describe()['table_energy_residual_max_J_kg']<1e-7
    assert model.config.physics.lh2_drag_coefficient==.5
