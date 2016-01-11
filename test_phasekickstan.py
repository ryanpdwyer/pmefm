from phasekickstan import update_models_dict
import numpy as np
from numpy.testing import assert_array_almost_equal
from nose import tools


def test_model_code_dict():
    model_code_dict = {'a': 'a', 'b': 'modified b', 'c': 'new model'}

    existing_dict = {
'a': ('a', hash('a')),
'b': ('b', hash('b')),
}

    expected_output = {
'a': 'a',
'b': 'modified b',
'c': 'new model'
}


    tools.eq_(update_models_dict(model_code_dict, existing_dict, test=True),
              expected_output)


def test_exp2df():
    df = np.array([1, 2])
    t = np.arange(1,4)
    tau = 1.

    df_expectec = np.array([[ 0.36787944,  1.13533528,  2.04978707],
                            [ 0.73575888,  2.27067057,  4.09957414]])

    






