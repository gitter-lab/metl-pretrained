import torch
import torch.hub

import metl.models as models
from metl.encode import DataEncoder, Encoding

UUID_URL_MAP = {
    "D72M9aEp": "https://uwmadison.box.com/shared/static/dj1b605pqmkep4eard45p75xvlk5nvpl.pt",
    "Nr9zCKpR": "https://uwmadison.box.com/shared/static/x03hzg0rvtomj3n47fkroahn7k38wu82.pt",
    "8gMPQJy4": "https://uwmadison.box.com/shared/static/2fyd0ecft0dlvfo29hvfina0fwcq0y46.pt",
    "Hr4GNHws": "https://uwmadison.box.com/shared/static/fveywo9t1jtbsl3qrhjcthgd3ltwfrnp.pt",

    "54etfaYj": "https://uwmadison.box.com/shared/static/rrefcranfmqrc9ghj6mu51abmkdb2mth.pt",
    "bcEoygY3": "https://uwmadison.box.com/shared/static/lgjj1sxctx1rkbuvp8nvzxuq5g5l2z1g.pt",

}

IDENT_UUID_MAP = {
    # the keys should be all lowercase
    "metl-g-20m-1d": "D72M9aEp",
    "metl-g-20m-3d": "Nr9zCKpR",
    "metl-l-2m-1d-gfp": "8gMPQJy4",
    "metl-l-2m-3d-gfp": "Hr4GNHws",
}


def download_checkpoint(uuid):
    ckpt = torch.hub.load_state_dict_from_url(UUID_URL_MAP[uuid],
                                              map_location="cpu", file_name=f"{uuid}.pt")
    state_dict = ckpt["state_dict"]
    hyper_parameters = ckpt["hyper_parameters"]

    return state_dict, hyper_parameters


def _get_data_encoding(hparams):
    if "encoding" in hparams and hparams["encoding"] == "int_seqs":
        encoding = Encoding.INT_SEQS
    elif "encoding" in hparams and hparams["encoding"] == "one_hot":
        encoding = Encoding.ONE_HOT
    elif (("encoding" in hparams and hparams["encoding"] == "auto") or "encoding" not in hparams) and \
            hparams["model_name"] in ["transformer_encoder"]:
        encoding = Encoding.INT_SEQS
    else:
        raise ValueError("Detected unsupported encoding in hyperparameters")

    return encoding


def load_model_and_data_encoder(state_dict, hparams):
    model = models.Model[hparams["model_name"]].cls(**hparams)
    model.load_state_dict(state_dict)

    data_encoder = DataEncoder(_get_data_encoding(hparams))

    return model, data_encoder


def get_from_uuid(uuid):
    if uuid in UUID_URL_MAP:
        state_dict, hparams = download_checkpoint(uuid)
        return load_model_and_data_encoder(state_dict, hparams)
    else:
        raise ValueError(f"UUID {uuid} not found in UUID_URL_MAP")


def get_from_ident(ident):
    ident = ident.lower()
    if ident in IDENT_UUID_MAP:
        state_dict, hparams = download_checkpoint(IDENT_UUID_MAP[ident])
        return load_model_and_data_encoder(state_dict, hparams)
    else:
        raise ValueError(f"Identifier {ident} not found in IDENT_UUID_MAP")


def get_from_checkpoint(ckpt_fn):
    ckpt = torch.load(ckpt_fn, map_location="cpu")
    state_dict = ckpt["state_dict"]
    hyper_parameters = ckpt["hyper_parameters"]
    return load_model_and_data_encoder(state_dict, hyper_parameters)
