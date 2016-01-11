from phasekickstan import update_models_dict
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
