from hydra import compose, initialize

from ppa_baseload_bess import Runner


def test_runner_smoke():
    """Runner(cfg).run() should complete without raising, end to end."""
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="config", overrides=["simulations=demo"])

    Runner(cfg).run()
