import torch
import torch.hub

import metl.models as models
from metl.encode import DataEncoder, Encoding
from metl.model_encoder import ModelEncoder

UUID_URL_MAP = {
    # global source models
    "D72M9aEp": "https://zenodo.org/records/14908509/files/METL-G-20M-1D-D72M9aEp.pt?download=1",
    "Nr9zCKpR": "https://zenodo.org/records/14908509/files/METL-G-20M-3D-Nr9zCKpR.pt?download=1",
    "auKdzzwX": "https://zenodo.org/records/14908509/files/METL-G-50M-1D-auKdzzwX.pt?download=1",
    "6PSAzdfv": "https://zenodo.org/records/14908509/files/METL-G-50M-3D-6PSAzdfv.pt?download=1",

    # local source models
    "8gMPQJy4": "https://zenodo.org/records/14908509/files/METL-L-2M-1D-GFP-8gMPQJy4.pt?download=1",
    "Hr4GNHws": "https://zenodo.org/records/14908509/files/METL-L-2M-3D-GFP-Hr4GNHws.pt?download=1",
    "8iFoiYw2": "https://zenodo.org/records/14908509/files/METL-L-2M-1D-DLG4_2022-8iFoiYw2.pt?download=1",
    "kt5DdWTa": "https://zenodo.org/records/14908509/files/METL-L-2M-3D-DLG4_2022-kt5DdWTa.pt?download=1",
    "DMfkjVzT": "https://zenodo.org/records/14908509/files/METL-L-2M-1D-GB1-DMfkjVzT.pt?download=1",
    "epegcFiH": "https://zenodo.org/records/14908509/files/METL-L-2M-3D-GB1-epegcFiH.pt?download=1",
    "kS3rUS7h": "https://zenodo.org/records/14908509/files/METL-L-2M-1D-GRB2-kS3rUS7h.pt?download=1",
    "X7w83g6S": "https://zenodo.org/records/14908509/files/METL-L-2M-3D-GRB2-X7w83g6S.pt?download=1",
    "UKebCQGz": "https://zenodo.org/records/14908509/files/METL-L-2M-1D-Pab1-UKebCQGz.pt?download=1",
    "2rr8V4th": "https://zenodo.org/records/14908509/files/METL-L-2M-3D-Pab1-2rr8V4th.pt?download=1",
    "PREhfC22": "https://zenodo.org/records/14908509/files/METL-L-2M-1D-TEM-1-PREhfC22.pt?download=1",
    "9ASvszux": "https://zenodo.org/records/14908509/files/METL-L-2M-3D-TEM-1-9ASvszux.pt?download=1",
    "HscFFkAb": "https://zenodo.org/records/14908509/files/METL-L-2M-1D-Ube4b-HscFFkAb.pt?download=1",
    "H48oiNZN": "https://zenodo.org/records/14908509/files/METL-L-2M-3D-Ube4b-H48oiNZN.pt?download=1",
    "CEMSx7ZC": "https://zenodo.org/records/14908509/files/METL-L-2M-1D-PTEN-CEMSx7ZC.pt?download=1",
    "PjxR5LW7": "https://zenodo.org/records/14908509/files/METL-L-2M-3D-PTEN-PjxR5LW7.pt?download=1",

    # metl bind source models
    "K6mw24Rg": "https://zenodo.org/records/14908509/files/METL-BIND-2M-3D-GB1-STANDARD-K6mw24Rg.pt?download=1",
    "Bo5wn2SG": "https://zenodo.org/records/14908509/files/METL-BIND-2M-3D-GB1-BINDING-Bo5wn2SG.pt?download=1",

    # finetuned models from GFP design experiment
    "YoQkzoLD": "https://zenodo.org/records/14908509/files/FT-METL-L-2M-1D-GFP-YoQkzoLD.pt?download=1",
    "PEkeRuxb": "https://zenodo.org/records/14908509/files/FT-METL-L-2M-3D-GFP-PEkeRuxb.pt?download=1",

    #  new finetuned GLOBAL models
    "4Rh3WCbG": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-1D-DLG4_2022-ABUNDANCE-4Rh3WCbG.pt?download=1",
    "4xbuC5y7": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-1D-DLG4_2022-BINDING-4xbuC5y7.pt?download=1",
    "dAndZfJ4": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-1D-GB1-dAndZfJ4.pt?download=1",
    "PeT2D92j": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-1D-GFP-PeT2D92j.pt?download=1",
    "HenDpDWe": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-1D-GRB2-ABUNDANCE-HenDpDWe.pt?download=1",
    "cvnycE5Q": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-1D-GRB2-BINDING-cvnycE5Q.pt?download=1",
    "ho54gxzv": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-1D-Pab1-ho54gxzv.pt?download=1",
    "UEuMtmfx": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-1D-PTEN-ABUNDANCE-UEuMtmfx.pt?download=1",
    "U3X8mSeT": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-1D-PTEN-ACTIVITY-U3X8mSeT.pt?download=1",
    "ELL4GGQq": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-1D-TEM-1-ELL4GGQq.pt?download=1",
    "BAWw23vW": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-1D-Ube4b-BAWw23vW.pt?download=1",
    "RBtqxzvu": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-3D-DLG4_2022-ABUNDANCE-RBtqxzvu.pt?download=1",
    "BuvxgE2x": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-3D-DLG4_2022-BINDING-BuvxgE2x.pt?download=1",
    "9vSB3DRM": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-3D-GB1-9vSB3DRM.pt?download=1",
    "6JBzHpkQ": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-3D-GFP-6JBzHpkQ.pt?download=1",
    "dDoCCvfr": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-3D-GRB2-ABUNDANCE-dDoCCvfr.pt?download=1",
    "jYesS9Ki": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-3D-GRB2-BINDING-jYesS9Ki.pt?download=1",
    "jhbL2FeB": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-3D-Pab1-jhbL2FeB.pt?download=1",
    "eJPPQYEW": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-3D-PTEN-ABUNDANCE-eJPPQYEW.pt?download=1",
    "4gqYnW6V": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-3D-PTEN-ACTIVITY-4gqYnW6V.pt?download=1",
    "K6BjsWXm": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-3D-TEM-1-K6BjsWXm.pt?download=1",
    "G9piq6WH": "https://zenodo.org/records/14908509/files/FT-METL-G-20M-3D-Ube4b-G9piq6WH.pt?download=1",

    # finetuned LOCAL models
    "RMFA6dnX": "https://zenodo.org/records/14908509/files/FT-METL-L-1D-DLG4_2022-ABUNDANCE-RMFA6dnX.pt?download=1",
    "YdzBYWHs": "https://zenodo.org/records/14908509/files/FT-METL-L-1D-DLG4_2022-BINDING-YdzBYWHs.pt?download=1",
    "Pgcseywk": "https://zenodo.org/records/14908509/files/FT-METL-L-1D-GB1-Pgcseywk.pt?download=1",
    "HaUuRwfE": "https://zenodo.org/records/14908509/files/FT-METL-L-1D-GFP-HaUuRwfE.pt?download=1",
    "VNpi9Zjt": "https://zenodo.org/records/14908509/files/FT-METL-L-1D-GRB2-ABUNDANCE-VNpi9Zjt.pt?download=1",
    "Z59BhUaE": "https://zenodo.org/records/14908509/files/FT-METL-L-1D-GRB2-BINDING-Z59BhUaE.pt?download=1",
    "TdjCzoQQ": "https://zenodo.org/records/14908509/files/FT-METL-L-1D-Pab1-TdjCzoQQ.pt?download=1",
    "64ncFxBR": "https://zenodo.org/records/14908509/files/FT-METL-L-1D-TEM-1-64ncFxBR.pt?download=1",
    "e9uhhnAv": "https://zenodo.org/records/14908509/files/FT-METL-L-1D-Ube4b-e9uhhnAv.pt?download=1",
    "oUScGeHo": "https://zenodo.org/records/14908509/files/FT-METL-L-2M-1D-PTEN-ABUNDANCE-oUScGeHo.pt?download=1",
    "m9UsG7dq": "https://zenodo.org/records/14908509/files/FT-METL-L-2M-1D-PTEN-ACTIVITY-m9UsG7dq.pt?download=1",
    "DhuasDEr": "https://zenodo.org/records/14908509/files/FT-METL-L-2M-3D-PTEN-ABUNDANCE-DhuasDEr.pt?download=1",
    "8Vi7ENcC": "https://zenodo.org/records/14908509/files/FT-METL-L-2M-3D-PTEN-ACTIVITY-8Vi7ENcC.pt?download=1",
    "V3uTtXVe": "https://zenodo.org/records/14908509/files/FT-METL-L-3D-DLG4_2022-ABUNDANCE-V3uTtXVe.pt?download=1",
    "iu6ZahPw": "https://zenodo.org/records/14908509/files/FT-METL-L-3D-DLG4_2022-BINDING-iu6ZahPw.pt?download=1",
    "UvMMdsq4": "https://zenodo.org/records/14908509/files/FT-METL-L-3D-GB1-UvMMdsq4.pt?download=1",
    "LWEY95Yb": "https://zenodo.org/records/14908509/files/FT-METL-L-3D-GFP-LWEY95Yb.pt?download=1",
    "PqBMjXkA": "https://zenodo.org/records/14908509/files/FT-METL-L-3D-GRB2-ABUNDANCE-PqBMjXkA.pt?download=1",
    "VwcRN6UB": "https://zenodo.org/records/14908509/files/FT-METL-L-3D-GRB2-BINDING-VwcRN6UB.pt?download=1",
    "5SjoLx3y": "https://zenodo.org/records/14908509/files/FT-METL-L-3D-Pab1-5SjoLx3y.pt?download=1",
    "PncvgiJU": "https://zenodo.org/records/14908509/files/FT-METL-L-3D-TEM-1-PncvgiJU.pt?download=1",
    "NfbZL7jK": "https://zenodo.org/records/14908509/files/FT-METL-L-3D-Ube4b-NfbZL7jK.pt?download=1"

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
    "metl-l-2m-1d-dlg4_2022": "8iFoiYw2",
    "metl-l-2m-3d-dlg4_2022": "kt5DdWTa",

    # GB1 local source models
    "metl-l-2m-1d-gb1": "DMfkjVzT",
    "metl-l-2m-3d-gb1": "epegcFiH",

    # GRB2 local source models
    "metl-l-2m-1d-grb2": "kS3rUS7h",
    "metl-l-2m-3d-grb2": "X7w83g6S",

    # Pab1 local source models
    "metl-l-2m-1d-pab1": "UKebCQGz",
    "metl-l-2m-3d-pab1": "2rr8V4th",

    # PTEN local source models
    "metl-l-2m-1d-pten": "CEMSx7ZC",
    "metl-l-2m-3d-pten": "PjxR5LW7",

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
    """
    Passing raw to this function dictates that the un-wrapped and unsafe version of the model and data encoder should be returned.
    These are what is saved onto zenodo. Strict decides how certain errors are handled. See ModelEncoder.validate_pdb for more detail.
    Indexing sets the expected index style of mutations passed into METL for predictions.
    """
    model = models.Model[hparams["model_name"]].cls(**hparams)
    model.load_state_dict(state_dict)

    data_encoder = DataEncoder(_get_data_encoding(hparams))

    if raw:
        return model, data_encoder
    else:
        return ModelEncoder(model, data_encoder, strict, indexing)


def get_from_uuid(uuid, strict=True, raw=False, indexing=0):
    """
    Strict is True here as models loaded from zenodo have matching PDB and wild types.

    Indexing is 0, as that is the default for METL. raw is always false by default as the preferred way to
    use METL (as an end-user) is with error handling enabled. 
    """

    if uuid in UUID_URL_MAP:
        state_dict, hparams = download_checkpoint(uuid)
        return load_model_and_data_encoder(state_dict, hparams, strict, raw, indexing)
    else:
        raise ValueError(f"UUID {uuid} not found in UUID_URL_MAP")


def get_from_ident(ident, strict=True, raw=False, indexing=0):
    """
    Strict is True here as models loaded from zenodo have matching PDB and wild types.

    Indexing is 0, as that is the default for METL. raw is always false by default as the preferred way to
    use METL (as an end-user) is with error handling enabled. 
    """

    ident = ident.lower()
    if ident in IDENT_UUID_MAP:
        state_dict, hparams = download_checkpoint(IDENT_UUID_MAP[ident])
        return load_model_and_data_encoder(state_dict, hparams, strict, raw, indexing)
    else:
        raise ValueError(f"Identifier {ident} not found in IDENT_UUID_MAP")


def get_from_checkpoint(ckpt_fn, strict=False, raw=False, indexing=0):
    """
    Strict is False here as checkpoint models are assumed to be fine-tuned, 
    and custom PDBs that don't exactly match the wild type may be more common here.

    Indexing is 0, as that is the default for METL. raw is always false by default as the preferred way to
    use METL (as an end-user) is with error handling enabled. 
    """
    
    ckpt = torch.load(ckpt_fn, map_location="cpu")
    state_dict = ckpt["state_dict"]
    hyper_parameters = ckpt["hyper_parameters"]
    return load_model_and_data_encoder(state_dict, hyper_parameters, strict, raw, indexing)
