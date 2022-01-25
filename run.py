import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


def target_var_id_mapping(x, y):
    k = 'l' if y.lower().replace('_', '').replace('-', '') == 'longwave' else 's'
    x = x.lower().replace('-', '_').replace('rates', 'rate').replace('_fluxes', '').replace('_flux', '')
    if x == 'heating_rate':
        return f'hr{k}c'
    elif x == 'upwelling':
        return f"r{k}uc"
    elif x == 'downwelling':
        return f"r{k}dc"
    else:
        raise ValueError(f"Combination {x} {y} not understood!")


OmegaConf.register_new_resolver("target_var_id", target_var_id_mapping)


@hydra.main(config_path="configs/", config_name="main_config.yaml")
def main(config: DictConfig):
    from climart.train import run_model
    return run_model(config)


if __name__ == "__main__":
    main()
