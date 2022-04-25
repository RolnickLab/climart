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


# -----------------------------------------------------

def no_clouds_exp(name):
    return name in ['pristine', 'clear-sky', 'clearsky', 'clear_sky', 'aerosols']


def get_flux_output_variables(target_type: str):
    if target_type.lower() == "shortwave":
        return OUT_SHORTWAVE_NOCLOUDS
    if target_type.lower() == "longwave":
        return OUT_LONGWAVE_NOCLOUDS
    if target_type.lower() == "shortwave+longwave":
        return OUT_SHORTWAVE_NOCLOUDS + OUT_LONGWAVE_NOCLOUDS
    raise ValueError(f" Unexpected arg {target_type} for target_type")


EXP_TYPES = ['pristine', 'clear_sky']
