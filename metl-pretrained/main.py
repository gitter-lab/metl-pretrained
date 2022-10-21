import torch
import torch.hub

import models
from encode import DataEncoder, Encoding


def download_checkpoint(model_name):
    url_map = {
        "kThmNaxC": "https://uwmadison.box.com/shared/static/eiyykekwkpxra77imi38j1z1g6nop7o0.pt",
        "VnsMZTkf": "https://uwmadison.box.com/shared/static/9tgzwxcrojj1pucv6scvxca2wklpxt9t.pt"
    }

    ckpt = torch.hub.load_state_dict_from_url(url_map[model_name], map_location="cpu", file_name=f"{model_name}.pt")
    state_dict = ckpt["state_dict"]
    hyper_parameters = ckpt["hyper_parameters"]

    return state_dict, hyper_parameters


def _get_data_encoding(hparams):
    if "encoding" in hparams and hparams["encoding"] == "int_seqs":
        encoding = Encoding.INT_SEQS
    elif "encoding" in hparams and hparams["encoding"] == "one_hot":
        encoding = Encoding.ONE_HOT
    elif "encoding" not in hparams and hparams["model_name"] in ["transformer_encoder"]:
        encoding = Encoding.INT_SEQS
    else:
        raise ValueError("Detected unsupported encoding in hyperparameters")

    return encoding


def load_model_and_data_encoder(model_name):
    state_dict, hparams = download_checkpoint(model_name)

    model = models.Model[hparams["model_name"]].cls(**hparams)
    model.load_state_dict(state_dict)

    data_encoder = DataEncoder(_get_data_encoding(hparams))

    return model, data_encoder


def kThmNaxC():
    return load_model_and_data_encoder("kThmNaxC")


def VnsMZTkf():
    return load_model_and_data_encoder("VnsMZTkf")
