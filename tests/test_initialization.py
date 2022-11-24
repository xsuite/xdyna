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

    assert_frame_equal(DA.survival_data[['x', 'y']], pd.DataFrame(data={'x':[1,2], 'y':[3,4]}) )
