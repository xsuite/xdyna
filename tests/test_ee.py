import json
from pathlib import Path
import pytest
import xobjects as xo
import xtrack as xt
import xpart as xp
import xdyna as xd

ENERGY = {'t': 182.5, 'z': 45.6}
EMITTANCE = {'t': {'X':1.46e-9, 'Y':2.9e-12, 'Z': 1.0}, 'z': {'X':0.71e-9, 'Y':1.42e-12, 'Z': 1.0}}
TURNS = {'t': 45, 'z': 2500}
TEST_DIR = Path(__file__).resolve().parent

@pytest.mark.parametrize("mode", ['t']) # use t mode for now, since z takes quite long
def test_simple_radial(mode):
    with open(TEST_DIR/'input'/f'tapered_{mode}_b1_thin.json', 'r', encoding='utf-8') as fid:
        loaded_dct = json.load(fid)
    line = xt.Line.from_dict(loaded_dct)

    context = xo.ContextCpu()

    ref_particle = xp.Particles(mass0=xp.ELECTRON_MASS_EV, q0=1, p0c=ENERGY[mode]*10**9, x=0, y=0)
    line.particle_ref = ref_particle
    tracker = xt.Tracker(_context=context, line=line)
    line.configure_radiation(model='mean')
#     tracker.configure_radiation(model='mean')

    DA = xd.DA(name=f'fcc_ee_{mode}',
               normalised_emittance=[EMITTANCE[mode]['X']*ref_particle.beta0[0]*ref_particle.gamma0[0],
                                     EMITTANCE[mode]['Y']*ref_particle.beta0[0]*ref_particle.gamma0[0]],
               max_turns=TURNS[mode],
               use_files=False)
    DA.generate_initial_radial(angles=1, r_min=2, r_max=20, r_step=4., delta=0.000)
    DA.line = line
    DA.track_job()


@pytest.mark.parametrize("mode", ['t']) # use t mode for now, since z takes quite long
def test_simple_grid(mode):
    with open(TEST_DIR/'input'/f'tapered_{mode}_b1_thin.json', 'r', encoding='utf-8') as fid:
        loaded_dct = json.load(fid)
    line = xt.Line.from_dict(loaded_dct)

    context = xo.ContextCpu()

    ref_particle = xp.Particles(mass0=xp.ELECTRON_MASS_EV, q0=1, p0c=ENERGY[mode]*10**9, x=0, y=0)
    line.particle_ref = ref_particle
    tracker = xt.Tracker(_context=context, line=line)
    line.configure_radiation(model='mean')
#     tracker.configure_radiation(model='mean')

    DA = xd.DA(name=f'fcc_ee_{mode}',
               normalised_emittance=[EMITTANCE[mode]['X']*ref_particle.beta0[0]*ref_particle.gamma0[0],
                                     EMITTANCE[mode]['Y']*ref_particle.beta0[0]*ref_particle.gamma0[0]],
               max_turns=TURNS[mode],
               use_files=False)
    DA.generate_initial_grid(x_min=0, x_max=20, x_step=10, y_min=0, y_max=20, y_step=10, delta=0.000)
    DA.line = line
    DA.track_job()
