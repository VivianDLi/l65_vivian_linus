"""
Main module to load and train the model. This should be the program entry
point.
"""

import graphein
import hydra
import lovely_tensors as lt
from omegaconf import DictConfig

from proteinworkshop import (
    register_custom_omegaconf_resolvers,
    utils,
)
from proteinworkshop.configs import config
from proteinworkshop.train import train_model

# adding proteinvirtual to path
import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from proteinvirtual import constants


graphein.verbose(False)
lt.monkey_patch()


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    version_base="1.3",
    config_path=str(constants.VIRTUAL_HYDRA_CONFIG_PATH),
    config_name="train",
)
def _main(cfg: DictConfig) -> None:
    """Load and validate the hydra config."""
    utils.extras(cfg)
    cfg = config.validate_config(cfg)
    train_model(cfg)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    register_custom_omegaconf_resolvers()
    _main()  # type: ignore
