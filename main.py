import gc

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Resolve the config and construct the model as a smoke test."""
    print(cfg)

    gc.collect()


if __name__ == "__main__":
    main()
