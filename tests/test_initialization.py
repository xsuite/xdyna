import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import xdyna as xd


SURV_COLUMNS = ['ang_xy', 'r_xy', 'nturns', 'x_norm_in', 'y_norm_in', 'px_norm_in',
       'py_norm_in', 'zeta_in', 'delta_in', 'x_out', 'y_out', 'px_out',
       'py_out', 'zeta_out', 'delta_out', 's_out', 'state', 'submitted',
       'finished']



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
    DA.set_coordinates(x=[1,2], y = [3,4], px=[5,6])

    assert_frame_equal(DA.survival_data[['x', 'y', 'px', 'py', 'delta']],
                       pd.DataFrame(data={
                                        'x':[1,2],
                                        'y':[3,4],
                                        'px':[5,6],
                                        'py':[0,0],
                                        'delta':[0,0]
                                            }) )


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
        ang_min=0, ang_max=90, angles=2, open_border=False
    )
    assert_frame_equal(DA.survival_data[['amplitude', 'angle']],
                       pd.DataFrame(data={'amplitude':[0.,2.,0.,2.], 'angle':[0.,0.,90.,90.]}) )


def test_pandas():

    DA = xd.DA(name='user_coordinates',
               normalised_emittance=[1,1],
               max_turns=2,
               use_files=False)
    DA.generate_initial_radial(
        r_min=0, r_max=2, r_step=2,
        ang_min=0, ang_max=90, angles=2,
    )

    assert_frame_equal(DA.survival_data,
                       DA.to_pandas())
    assert all(elem in DA.to_pandas(full=True).columns for elem in SURV_COLUMNS)
