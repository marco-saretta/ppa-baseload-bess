import gc

import hydra
from omegaconf import DictConfig

from ppa_baseload_bess import Runner


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Resolve the config and run the pipeline as a smoke test."""

    runner = Runner(cfg)
    runner.run()
    gc.collect()


if __name__ == "__main__":
    main()
