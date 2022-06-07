import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
from climart.utils.utils import target_var_id_mapping

dotenv.load_dotenv(override=True)
OmegaConf.register_new_resolver("target_var_id", target_var_id_mapping)


@hydra.main(config_path="configs/", config_name="main_config.yaml")
def main(config: DictConfig):
    from climart.train import run_model
    return run_model(config)


if __name__ == "__main__":
    main()
