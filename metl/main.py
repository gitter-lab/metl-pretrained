import torch
import torch.hub

import metl.models as models
from metl.encode import DataEncoder, Encoding
from metl.model_encoder import ModelEncoder

UUID_URL_MAP = {
    # global source models
    "D72M9aEp": "https://zenodo.org/records/11051645/files/METL-G-20M-1D-D72M9aEp.pt?download=1",
    "Nr9zCKpR": "https://zenodo.org/records/11051645/files/METL-G-20M-3D-Nr9zCKpR.pt?download=1",
    "auKdzzwX": "https://zenodo.org/records/11051645/files/METL-G-50M-1D-auKdzzwX.pt?download=1",
    "6PSAzdfv": "https://zenodo.org/records/11051645/files/METL-G-50M-3D-6PSAzdfv.pt?download=1",

    # local source models
    "8gMPQJy4": "https://zenodo.org/records/11051645/files/METL-L-2M-1D-GFP-8gMPQJy4.pt?download=1",
    "Hr4GNHws": "https://zenodo.org/records/11051645/files/METL-L-2M-3D-GFP-Hr4GNHws.pt?download=1",
    "8iFoiYw2": "https://zenodo.org/records/11051645/files/METL-L-2M-1D-DLG4_2022-8iFoiYw2.pt?download=1",
    "kt5DdWTa": "https://zenodo.org/records/11051645/files/METL-L-2M-3D-DLG4_2022-kt5DdWTa.pt?download=1",
    "DMfkjVzT": "https://zenodo.org/records/11051645/files/METL-L-2M-1D-GB1-DMfkjVzT.pt?download=1",
    "epegcFiH": "https://zenodo.org/records/11051645/files/METL-L-2M-3D-GB1-epegcFiH.pt?download=1",
    "kS3rUS7h": "https://zenodo.org/records/11051645/files/METL-L-2M-1D-GRB2-kS3rUS7h.pt?download=1",
    "X7w83g6S": "https://zenodo.org/records/11051645/files/METL-L-2M-3D-GRB2-X7w83g6S.pt?download=1",
    "UKebCQGz": "https://zenodo.org/records/11051645/files/METL-L-2M-1D-Pab1-UKebCQGz.pt?download=1",
    "2rr8V4th": "https://zenodo.org/records/11051645/files/METL-L-2M-3D-Pab1-2rr8V4th.pt?download=1",
    "PREhfC22": "https://zenodo.org/records/11051645/files/METL-L-2M-1D-TEM-1-PREhfC22.pt?download=1",
    "9ASvszux": "https://zenodo.org/records/11051645/files/METL-L-2M-3D-TEM-1-9ASvszux.pt?download=1",
    "HscFFkAb": "https://zenodo.org/records/11051645/files/METL-L-2M-1D-Ube4b-HscFFkAb.pt?download=1",
    "H48oiNZN": "https://zenodo.org/records/11051645/files/METL-L-2M-3D-Ube4b-H48oiNZN.pt?download=1",

    # metl bind source models
    "K6mw24Rg": "https://zenodo.org/records/11051645/files/METL-BIND-2M-3D-GB1-STANDARD-K6mw24Rg.pt?download=1",
    "Bo5wn2SG": "https://zenodo.org/records/11051645/files/METL-BIND-2M-3D-GB1-BINDING-Bo5wn2SG.pt?download=1",

    # finetuned models from GFP design experiment
    "YoQkzoLD": "https://zenodo.org/records/11051645/files/FT-METL-L-2M-1D-GFP-YoQkzoLD.pt?download=1",
    "PEkeRuxb": "https://zenodo.org/records/11051645/files/FT-METL-L-2M-3D-GFP-PEkeRuxb.pt?download=1",

}

IDENT_UUID_MAP = {
    # the keys should be all lowercase
    "metl-g-20m-1d": "D72M9aEp",
    "metl-g-20m-3d": "Nr9zCKpR",
    "metl-g-50m-1d": "auKdzzwX",
    "metl-g-50m-3d": "6PSAzdfv",

    # GFP local source models
    "metl-l-2m-1d-gfp": "8gMPQJy4",
    "metl-l-2m-3d-gfp": "Hr4GNHws",

    # DLG4 local source models
    "metl-l-2m-1d-dlg4": "8iFoiYw2",
    "metl-l-2m-3d-dlg4": "kt5DdWTa",

    # GB1 local source models
    "metl-l-2m-1d-gb1": "DMfkjVzT",
    "metl-l-2m-3d-gb1": "epegcFiH",

    # GRB2 local source models
    "metl-l-2m-1d-grb2": "kS3rUS7h",
    "metl-l-2m-3d-grb2": "X7w83g6S",

    # Pab1 local source models
    "metl-l-2m-1d-pab1": "UKebCQGz",
    "metl-l-2m-3d-pab1": "2rr8V4th",

    # TEM-1 local source models
    "metl-l-2m-1d-tem-1": "PREhfC22",
    "metl-l-2m-3d-tem-1": "9ASvszux",

    # Ube4b local source models
    "metl-l-2m-1d-ube4b": "HscFFkAb",
    "metl-l-2m-3d-ube4b": "H48oiNZN",

    # METL-Bind for GB1
    "metl-bind-2m-3d-gb1-standard": "K6mw24Rg",
    "metl-bind-2m-3d-gb1-binding": "Bo5wn2SG",

    # GFP design models, giving them an ident
    "metl-l-2m-1d-gfp-ft-design": "YoQkzoLD",
    "metl-l-2m-3d-gfp-ft-design": "PEkeRuxb",

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


def load_model_and_data_encoder(state_dict, hparams, strict, raw, indexing):
    model = models.Model[hparams["model_name"]].cls(**hparams)
    model.load_state_dict(state_dict)

    data_encoder = DataEncoder(_get_data_encoding(hparams))

    if raw:
        return model, data_encoder
    else:
        return ModelEncoder(model, data_encoder, strict, indexing)


def get_from_uuid(uuid, strict=True, raw=False, indexing=0):
    if uuid in UUID_URL_MAP:
        state_dict, hparams = download_checkpoint(uuid)
        return load_model_and_data_encoder(state_dict, hparams, strict, raw, indexing)
    else:
        raise ValueError(f"UUID {uuid} not found in UUID_URL_MAP")


def get_from_ident(ident, strict=True, raw=False, indexing=0):
    ident = ident.lower()
    if ident in IDENT_UUID_MAP:
        state_dict, hparams = download_checkpoint(IDENT_UUID_MAP[ident])
        return load_model_and_data_encoder(state_dict, hparams, strict, raw, indexing)
    else:
        raise ValueError(f"Identifier {ident} not found in IDENT_UUID_MAP")


def get_from_checkpoint(ckpt_fn, strict=False, raw=False, indexing=0):
    ckpt = torch.load(ckpt_fn, map_location="cpu")
    state_dict = ckpt["state_dict"]
    hyper_parameters = ckpt["hyper_parameters"]
    return load_model_and_data_encoder(state_dict, hyper_parameters, strict, raw, indexing)
