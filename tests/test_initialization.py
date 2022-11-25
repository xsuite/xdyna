import pytest
import xdyna as xd
import pandas as pd
from pandas.testing import assert_frame_equal


def test_mismatch_user_coordinates():

    DA = xd.DA(name='user_coordinates',
               normalised_emittance=[1,1],
               max_turns=2,
               use_files=False)
    with pytest.raises(AssertionError):
        DA.set_coordinates(x=[1], y = [1,2])


def test_user_coordinates():

    DA = xd.DA(name='user_coordinates',
               normalised_emittance=[1,1],
               max_turns=2,
               use_files=False)
    DA.set_coordinates(x=[1,2], y = [3,4])

    assert_frame_equal(DA.survival_data[['x', 'y']],
                       pd.DataFrame(data={'x':[1,2], 'y':[3,4]}) )


def test_xy_grid():

    DA = xd.DA(name='user_coordinates',
               normalised_emittance=[1,1],
               max_turns=2,
               use_files=False)
    DA.generate_initial_grid(
        x_min=0, x_max=2, x_step=2,
        y_min=0, y_max=2, y_step=2,
    )
    assert_frame_equal(DA.survival_data[['x', 'y']],
                       pd.DataFrame(data={'x':[0.,2.,0.,2.], 'y':[0.,0.,2.,2.]}) )


def test_radial_grid():

    DA = xd.DA(name='user_coordinates',
               normalised_emittance=[1,1],
               max_turns=2,
               use_files=False)
    DA.generate_initial_radial(
        r_min=0, r_max=2, r_step=2,
        ang_min=0, ang_max=90, angles=2,
    )
    assert_frame_equal(DA.survival_data[['amplitude', 'angle']],
                       pd.DataFrame(data={'amplitude':[0.,2.,0.,2.], 'angle':[0.,0.,90.,90.]}) )
