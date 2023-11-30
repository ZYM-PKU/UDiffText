from .encoders.modules import GeneralConditioner, DualConditioner

UNCONDITIONAL_CONFIG = {
    "target": "sgm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
