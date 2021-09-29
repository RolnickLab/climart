from rtml.data_wrangling.data_variables import INPUT_VARS_CLOUDS, INPUT_VARS_AEROSOLS, _ALL_INPUT_VARS


def exp_type_subset_vars_test():
    for k in INPUT_VARS_CLOUDS:
        assert k in _ALL_INPUT_VARS, f"Cloud var {k} was expected to be in _ALL_INPUT_VARS."
    for k in INPUT_VARS_AEROSOLS:
        assert k in _ALL_INPUT_VARS, f"Aerosol var {k} was expected to be in _ALL_INPUT_VARS."
