import logging
from typing import List

_ALL_INPUT_VARS = [
    'shtj',
    'tfrow',
    'shj',
    'dshj',
    'dz',
    'tlayer',
    'ozphs',
    'qc',
    'co2rox',
    'ch4rox',
    'n2orox',
    'f11rox',
    'f12rox',
    'ccld',
    'rhc',
    'anu',
    'eta',
    'aerin',
    'sw_ext_sa',
    'sw_ssa_sa',
    'sw_g_sa',
    'lw_abs_sa',
    'pressg',
    'gtrow',
    'oztop',
    'cszrow',
    'vtaurow',
    'troprow',
    'emisrow',
    'cldtrol',
    'ncldy',
    'salbrol',
    'csalrol',
    'emisrot',
    'gtrot',
    'farerot',
    'salbrot',
    'csalrot',
    'rel_sub',
    'rei_sub',
    'clw_sub',
    'cic_sub',
    'layer_pressure',
    'level_pressure',
    'layer_thickness',
    'x_cord',
    'y_cord',
    'z_cord',
    'temp_diff',
    'height'
]


NEVER_USE_VARS = [
    "iseed", 'f113rox', 'f114rox'  # the last two are constant 0
]

# -------------------------------------- OUTPUTS/TARGET VARIABLES
LW_HEATING_RATE = 'hrlc'
SW_HEATING_RATE = 'hrsc'

OUT_HEATING_RATE_CLOUDS = [
    'hrl',  # heating rate (long-wave)
    'hrs'  # heating rate (short-wave)
]

OUT_HEATING_RATE_NOCLOUDS = [
    LW_HEATING_RATE,  # heating rate (long-wave)
    SW_HEATING_RATE  # heating rate (short-wave)
]

OUT_SHORTWAVE_CLOUDS = [
    'rsd',  # solar flux down
    'rsu',  # solar flux up
]

OUT_SHORTWAVE_NOCLOUDS = [
    'rsdc',  # solar flux down
    'rsuc',  # solar flux up
]

OUT_LONGWAVE_CLOUDS = [
    "rld",  # thermal flux down
    'rlu',  # thermal flux up
]

OUT_LONGWAVE_NOCLOUDS = [
    "rldc",  # thermal flux down
    'rluc',  # thermal flux up
]

_ALL_OUTPUT_VARS = OUT_SHORTWAVE_NOCLOUDS + OUT_LONGWAVE_NOCLOUDS + OUT_HEATING_RATE_NOCLOUDS
_ALL_VARS = _ALL_INPUT_VARS + _ALL_OUTPUT_VARS

INPUT_VARS_CLOUDS = [
    'ccld',  # Cloud amount profile
    'anu',  # Cloud water content horizontal variability parameter
    'eta',  # Fraction black carbon in liquid cloud droplets
    'vtaurow',  # Vertically integrated optical thickness at 550 nm for stratospheric aerosols
    'troprow',  # Layer index of the tropopause
    'cldtrol',  # Total vertically projected cloud fraction
    'ncldy',  # Number of cloudy subcolumns in CanAM grid
    'rel_sub',  # Liquid cloud effective radius for subcolumns in CanAM grid
    'rei_sub',  # Ice cloud effective radius for subcolumns in CanAM grid
    'clw_sub',  # Liquid cloud water path for subcolumns in CanAM grid
    'cic_sub'
]

INPUT_VARS_AEROSOLS = [
    'rhc',  # Relative humidity
    'aerin',  # Relative humidity
    'sw_ext_sa',  # Cloud water content horizontal variability parameter
    'sw_ssa_sa',  # solar flux up
    'sw_g_sa',  # heating rate (long-wave?)
    'lw_abs_sa'  # heating rate (short-wave?)
]

DONT_NORMALIZE = ['x_cord', 'y_cord', 'z_cord']


def get_all_vars(var_type: str,
                 exp_type: str) -> List[str]:
    var_type = var_type.lower()
    exp_type = exp_type.lower().replace('_', '').replace('-', '')
    assert var_type in ['input', 'output'], "Argument var_type must be 'input' or 'output'"
    all_vars = _ALL_INPUT_VARS.copy()
    if var_type == 'input':
        if exp_type in ['all', 'clouds', 'allsky']:
            return all_vars
        elif exp_type in ['clearsky', 'aerosols']:
            clearsky_vars = [
                var for var in all_vars
                if var not in INPUT_VARS_CLOUDS
            ]
            return clearsky_vars
        elif exp_type == 'pristine':
            pristine_vars = [
                var for var in all_vars
                if (var not in INPUT_VARS_CLOUDS and var not in INPUT_VARS_AEROSOLS)
            ]
            return pristine_vars
        else:
            raise ValueError()
    else:
        if exp_type == 'all':
            return OUT_HEATING_RATE_CLOUDS + OUT_LONGWAVE_CLOUDS + OUT_SHORTWAVE_CLOUDS + \
                   OUT_HEATING_RATE_NOCLOUDS + OUT_LONGWAVE_NOCLOUDS + OUT_SHORTWAVE_NOCLOUDS
        elif exp_type in ['clouds', 'allsky']:
            return OUT_HEATING_RATE_CLOUDS + OUT_LONGWAVE_CLOUDS + OUT_SHORTWAVE_CLOUDS
        elif no_clouds_exp(exp_type):
            return OUT_HEATING_RATE_NOCLOUDS + OUT_LONGWAVE_NOCLOUDS + OUT_SHORTWAVE_NOCLOUDS
        else:
            raise ValueError()


# -----------------------------------------------------

def no_clouds_exp(name):
    return name in ['pristine', 'clear-sky', 'clearsky', 'clear_sky', 'aerosols']


def use_log_scaling(name):
    return name in ['dz', 'pressg', 'ozphs', 'oztop', 'qc', 'tlayer', 'layer_pressure', 'level_pressure']


def use_surface_variables(name):
    return name in ['salbrol', 'csalrol', 'emisrot', 'gtrot', 'farerot', 'salbrot', 'csalrot']


def use_output_variable(var_name, exp='pristine', target='shortwave'):
    var_name = var_name.lower().strip()
    target = target.lower().strip()
    exp = exp.lower().strip()

    if var_name in NEVER_USE_VARS:
        return False
    elif no_clouds_exp(exp):
        if target == 'shortwave':
            return var_name in OUT_SHORTWAVE_NOCLOUDS
        elif target == 'longwave':
            return var_name in OUT_LONGWAVE_NOCLOUDS
        elif target == 'heating_rate':
            return var_name in OUT_HEATING_RATE_NOCLOUDS
    else:
        if target == 'shortwave':
            return var_name in OUT_SHORTWAVE_CLOUDS
        elif target == 'longwave':
            return var_name in OUT_LONGWAVE_CLOUDS
        elif target == 'heating_rate':
            return var_name in OUT_HEATING_RATE_CLOUDS
    raise ValueError(f"{target} not known as target variable!")


def use_input_variable(var_name, exp='pristine'):
    var_name = var_name.lower().strip()
    exp = exp.lower().strip()

    if var_name in NEVER_USE_VARS:
        return False

    elif var_name in INPUT_VARS_AEROSOLS:
        return False if exp == 'pristine' else True

    elif var_name in INPUT_VARS_CLOUDS:
        return False if no_clouds_exp(exp) else True

    else:
        print('DEFAULTING TO USING THE VARIABLE', var_name)
        return True
        raise ValueError(f"Unknown variable name {var_name}!")


def reorder_input_variables(variable_names_list: List[str]):
    """ Reorders the list order, such that similar/equivalent variables come subsequently to each other"""
    tmp_list = variable_names_list.copy()
    ordered_vars = []
    # GLOBALS, Layers, levels
    ordered_vars += ['cszrow', 'shj', 'shtj']
    ordered_vars += ['gtrow', 'tlayer', 'tfrow']
    if 'layer_pressure' in variable_names_list:
        ordered_vars += ['pressg', 'layer_pressure', 'level_pressure']
    ordered_vars += ['oztop', 'ozphs']
    ordered_vars += ['qc']
    ordered_vars += ['dz']

    end_vars = []
    if 'x_cord' in variable_names_list:
        end_vars += ['x_cord', 'y_cord', 'z_cord']

    in_clouds = [v for v in INPUT_VARS_CLOUDS if v in tmp_list]
    in_aerosol = [v for v in INPUT_VARS_AEROSOLS if v in tmp_list]

    for var in ordered_vars + end_vars + in_clouds + in_aerosol:
        print('--->', var)
        logging.info(var, '<<<<<<<<<')
        tmp_list.remove(var)

    ordered_vars += tmp_list
    ordered_vars += end_vars

    ordered_vars += in_aerosol
    ordered_vars += in_clouds
    return ordered_vars


'''
OUTPUT_VARS_CLOUDS = [
    "rld",  # thermal flux down
    'rlu',  # thermal flux up
    'rsd',  # solar flux down
    'rsu',  # solar flux up
    'hrl',  # heating rate (long-wave?)
    'hrs'  # heating rate (short-wave?)
]

OUTPUT_VARS_NO_CLOUDS = [   # for pristine or clear-sky conditions
    # "rldc",  # thermal flux down
    # 'rluc',  # thermal flux up
    'rsdc',  # solar flux down
    'rsuc',  # solar flux up
    # 'hrlc',  # heating rate (long-wave?)
    # 'hrsc'  # heating rate (short-wave?)
]
'''

# ncldy=ncldy*0; aerin=aerin*0.0f; sw_ext_sa=sw_ext_sa*0.0f; sw_ssa_sa=sw_ssa_sa*0.0f;
# sw_g_sa=sw_g_sa*0.0f; lw_abs_sa=lw_abs_sa*0.0f; ccld=ccld*0.0f; cldtrol=cldtrol*0.0f;
# clw_sub=clw_sub*0.0f; cic_sub=cic_sub*0.0f; rel_sub=rel_sub*0.0f;rei_sub=rei_sub*0.0f'
# The following vars need to be zeroed out before passing them to the RTE solver for generating corresponding outputs
input_variables_to_zero_out_for_exp = {
    'pristine': [
        'ncldy', 'aerin', 'sw_ext_sa', 'sw_ssa_sa',
        'sw_g_sa', 'lw_abs_sa', 'ccld', 'cldtrol',
        'clw_sub', 'cic_sub', 'rel_sub', 'rei_sub'
    ],
    'clear_sky': [
        'ncldy', 'ccld', 'cldtrol',
        'clw_sub', 'cic_sub', 'rel_sub', 'rei_sub'
    ],
    'all_sky': []
}

# The following vars need not be used for the corresponding experiments as inputs
input_variables_to_drop_for_exp = {
    'pristine':
        ['iseed'] + list(
            set(input_variables_to_zero_out_for_exp['pristine'] + INPUT_VARS_AEROSOLS + INPUT_VARS_CLOUDS)),
    'clear_sky':
        ['iseed'] + list(set(input_variables_to_zero_out_for_exp['clear_sky'] + INPUT_VARS_CLOUDS)),
    'all_sky':
        ['iseed']
}

output_variables_to_drop_for_exp = {
    'pristine': OUT_SHORTWAVE_CLOUDS + OUT_LONGWAVE_CLOUDS + OUT_HEATING_RATE_CLOUDS,
    'clear_sky': OUT_SHORTWAVE_CLOUDS + OUT_LONGWAVE_CLOUDS + OUT_HEATING_RATE_CLOUDS,
    'all_sky': OUT_SHORTWAVE_NOCLOUDS + OUT_LONGWAVE_NOCLOUDS + OUT_HEATING_RATE_NOCLOUDS
}

EXP_TYPES = ['pristine', 'clear_sky'] #, 'all_sky']
