import copy
import os
from typing import List

import omegaconf
import pytest
from hydra.utils import instantiate

from proteinworkshop import register_custom_omegaconf_resolvers
from proteinvirtual import constants

ENCODERS: List[str] = [
    "gnn_geo_hetero_hierarchy",
    "gnn_nongeo_hetero_hierarchy",
    "schnet_hierarchy",
    "gnn_nongeo_hetero",
    "schnet",
]

FEATURES = os.listdir(constants.VIRTUAL_HYDRA_CONFIG_PATH / "features")

register_custom_omegaconf_resolvers()


def test_instantiate_encoders():
    for encoder in ENCODERS:
        config_path = (
            constants.VIRTUAL_HYDRA_CONFIG_PATH / "encoder" / f"{encoder}.yaml"
        )
        cfg = omegaconf.OmegaConf.create()
        cfg.encoder = omegaconf.OmegaConf.load(config_path)
        cfg.features = omegaconf.OmegaConf.load(
            constants.VIRTUAL_HYDRA_CONFIG_PATH
            / "features"
            / (
                "ca_nongeo_hetero_hierarchy.yaml"
                if "hierarchy" in config_path
                else "ca_nongeo_hetero.yaml"
            )
        )
        cfg.task = omegaconf.OmegaConf.load(
            constants.HYDRA_CONFIG_PATH
            / "task"
            / "multiclass_graph_classification.yaml"
        )

        enc = instantiate(cfg.encoder)

        assert enc, f"Encoder {encoder} not instantiated!"


@pytest.mark.skip(reason="Too slow for GitHub Actions. Works locally.")
def test_encoder_forward_pass(example_batch):
    for encoder in ENCODERS:
        for feature in FEATURES:
            encoder_config_path = (
                constants.VIRTUAL_HYDRA_CONFIG_PATH
                / "encoder"
                / f"{encoder}.yaml"
            )
            feature_config_path = (
                constants.VIRTUAL_HYDRA_CONFIG_PATH / "features" / feature
            )

            cfg = omegaconf.OmegaConf.create()
            cfg.encoder = omegaconf.OmegaConf.load(encoder_config_path)
            cfg.features = omegaconf.OmegaConf.load(feature_config_path)
            cfg.task = omegaconf.OmegaConf.load(
                constants.HYDRA_CONFIG_PATH
                / "task"
                / "multiclass_graph_classification.yaml"
            )

            enc = instantiate(cfg.encoder)
            featuriser = instantiate(cfg.features)

            batch = featuriser(copy.copy(example_batch))
            print(example_batch)
            print(enc)
            out = enc(batch)
            assert out
            assert isinstance(out, dict)
            assert "node_embedding" in out
            assert "graph_embedding" in out
