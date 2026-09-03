"""Test run entrypoint.

    uv run python main.py
"""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Print the resolved config as a smoke test."""
    print(f"message: {cfg.message}")
    print(f"seed: {cfg.seed}")
    print("---")
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
