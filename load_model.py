import torch

from model import Model
from ckpt_util import get_ckpt_path


def from_pretrained(name):
    cifar10_cfg = {
        "resolution": 32,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": (1, 2, 2, 2),
        "num_res_blocks": 2,
        "attn_resolutions": (16,),
        "dropout": 0.1,
    }
    lsun_cfg = {
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": (1, 1, 2, 2, 4, 4),
        "num_res_blocks": 2,
        "attn_resolutions": (16,),
        "dropout": 0.0,
    }

    model_config_map = {
        "cifar10": cifar10_cfg,
        "lsun_bedroom": lsun_cfg,
        "lsun_cat": lsun_cfg,
        "lsun_church": lsun_cfg,
    }

    diffusion_config = {
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "num_diffusion_timesteps": 1000,
    }
    model_var_type_map = {
        "cifar10": "fixedlarge",
        "lsun_bedroom": "fixedsmall",
        "lsun_cat": "fixedsmall",
        "lsun_church": "fixedsmall",
    }
    ema = name.startswith("ema_")
    basename = name[len("ema_"):] if ema else name
    diffusion_config["model_var_type"] = model_var_type_map[basename]

    print("Instantiating")
    model = Model(**model_config_map[basename])

    ckpt = get_ckpt_path(name)
    print("Loading checkpoint {}".format(ckpt))
    model.load_state_dict(torch.load(ckpt))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    return model


def load_model(name):
    model = from_pretrained(name)