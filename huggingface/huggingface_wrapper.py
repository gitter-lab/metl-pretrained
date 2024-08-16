from transformers import PretrainedConfig, PreTrainedModel
from enum import Enum, auto
import enum
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear, Dropout, LayerNorm
import math
from argparse import ArgumentParser
from typing import List, Tuple, Optional, Union
from os.path import basename, dirname, join, isfile
from scipy.spatial.distance import cdist
import collections
import torch.nn.functional as F
import time
import networkx as nx
import copy
from biopandas.pdb import PandasPdb
import os

#replace model.Model with Model
#replace models.get_activation_fn with get_activation_fn
#replace models.reset_parameters_helper with reset_parameters_helper
#replace ra.RelativeTransformerEncoder with RelativeTransformerEncoder
#replace ra.RelativeTransformerEncoderLayer with RelativeTransformerEncoderLayer
#replace structure.cbeta_distance_matrix with cbeta_distance_matrix
#replace structure.dist_thresh_graph with dist_thresh_graph 

# Encode
""" Encodes data in different formats """
# from enum import Enum, auto

# import numpy as np


class Encoding(Enum):
    INT_SEQS = auto()
    ONE_HOT = auto()


class DataEncoder:
    chars = ["*", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    num_chars = len(chars)
    mapping = {c: i for i, c in enumerate(chars)}

    def __init__(self, encoding: Encoding = Encoding.INT_SEQS):
        self.encoding = encoding

    def _encode_from_int_seqs(self, seq_ints):
        if self.encoding == Encoding.INT_SEQS:
            return seq_ints
        elif self.encoding == Encoding.ONE_HOT:
            one_hot = np.eye(self.num_chars)[seq_ints]
            return one_hot.astype(np.float32)

    def encode_sequences(self, char_seqs):
        seq_ints = []
        for char_seq in char_seqs:
            int_seq = [self.mapping[c] for c in char_seq]
            seq_ints.append(int_seq)
        seq_ints = np.array(seq_ints).astype(int)
        return self._encode_from_int_seqs(seq_ints)

    def encode_variants(self, wt, variants):
        # convert wild type seq to integer encoding
        wt_int = np.zeros(len(wt), dtype=np.uint8)
        for i, c in enumerate(wt):
            wt_int[i] = self.mapping[c]

        # tile the wild-type seq
        seq_ints = np.tile(wt_int, (len(variants), 1))

        for i, variant in enumerate(variants):
            # special handling if we want to encode the wild-type seq (it's already correct!)
            if variant == "_wt":
                continue

            # variants are a list of mutations [mutation1, mutation2, ....]
            variant = variant.split(",")
            for mutation in variant:
                # mutations are in the form <original char><position><replacement char>
                position = int(mutation[1:-1])
                replacement = self.mapping[mutation[-1]]
                seq_ints[i, position] = replacement

        seq_ints = seq_ints.astype(int)
        return self._encode_from_int_seqs(seq_ints)

## main

# import torch
# import torch.hub

# import metl.models as models
# from metl.encode import DataEncoder, Encoding

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


def load_model_and_data_encoder(state_dict, hparams):
    model = Model[hparams["model_name"]].cls(**hparams)
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

### models

# import collections
# import math
# from argparse import ArgumentParser
# import enum
# from os.path import isfile
# from typing import List, Tuple, Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor

# import metl.relative_attention as ra


def reset_parameters_helper(m: nn.Module):
    """ helper function for resetting model parameters, meant to be used with model.apply() """

    # the PyTorch MultiHeadAttention has a private function _reset_parameters()
    # other layers have a public reset_parameters()... go figure
    reset_parameters = getattr(m, "reset_parameters", None)
    reset_parameters_private = getattr(m, "_reset_parameters", None)

    if callable(reset_parameters) and callable(reset_parameters_private):
        raise RuntimeError("Module has both public and private methods for resetting parameters. "
                           "This is unexpected... probably should just call the public one.")

    if callable(reset_parameters):
        m.reset_parameters()

    if callable(reset_parameters_private):
        m._reset_parameters()


class SequentialWithArgs(nn.Sequential):
    def forward(self, x, **kwargs):
        for module in self:
            if isinstance(module, RelativeTransformerEncoder) or isinstance(module, SequentialWithArgs):
                # for relative transformer encoders, pass in kwargs (pdb_fn)
                x = module(x, **kwargs)
            else:
                # for all modules, don't pass in kwargs
                x = module(x)
        return x


class PositionalEncoding(nn.Module):
    # originally from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    # they have since updated their implementation, but it is functionally equivalent
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # note the implementation on Pytorch's website expects [seq_len, batch_size, embedding_dim]
        # however our data is in [batch_size, seq_len, embedding_dim] (i.e. batch_first)
        # fixed by changing pe = pe.unsqueeze(0).transpose(0, 1) to pe = pe.unsqueeze(0)
        # also down below, changing our indexing into the position encoding to reflect new dimensions
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, **kwargs):
        # note the implementation on Pytorch's website expects [seq_len, batch_size, embedding_dim]
        # however our data is in [batch_size, seq_len, embedding_dim] (i.e. batch_first)
        # fixed by changing x = x + self.pe[:x.size(0)] to x = x + self.pe[:, :x.size(1), :]
        # x = x + self.pe[:x.size(0), :]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ScaledEmbedding(nn.Module):
    # https://pytorch.org/tutorials/beginner/translation_transformer.html
    # a helper function for embedding that scales by sqrt(d_model) in the forward()
    # makes it, so we don't have to do the scaling in the main AttnModel forward()

    # todo: be aware of embedding scaling factor
    # regarding the scaling factor, it's unclear exactly what the purpose is and whether it is needed
    # there are several theories on why it is used, and it shows up in all the transformer reference implementations
    # https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod
    #   1. Has something to do with weight sharing between the embedding and the decoder output
    #   2. Scales up the embeddings so the signal doesn't get overwhelmed when adding the absolute positional encoding
    #   3. It cancels out with the scaling factor in scaled dot product attention, and helps make the model robust
    #      to the choice of embedding_len
    #   4. It's not actually needed

    # Regarding #1, not really sure about this. In section 3.4 of attention is all you need,
    # that's where they state they multiply the embedding weights by sqrt(d_model), and the context is that they
    # are sharing the same weight matrix between the two embedding layers and the pre-softmax linear transformation.
    # there may be a reason that we want those weights scaled differently for the embedding layers vs. the linear
    # transformation. It might have something to do with the scale at which embedding weights are initialized
    # is more appropriate for the decoder linear transform vs how they are used in the attention function. Might have
    # something to do with computing the correct next-token probabilities. Overall, I'm really not sure about this,
    # but we aren't using a decoder anyway. So if this is the reason, then we don't need to perform the multiply.

    # Regarding #2, it seems like in one implementation of transformers (fairseq), the sinusoidal positional encoding
    # has a range of (-1.0, 1.0), but the word embedding are initialized with mean 0 and s.d embedding_dim ** -0.5,
    # which for embedding_dim=512, is a range closer to (-0.10, 0.10). Thus, the positional embedding would overwhelm
    # the word embeddings when they are added together. The scaling factor increases the signal of the word embeddings.
    # for embedding_dim=512, it scales word embeddings by 22, increasing range of the word embeddings to (-2.2, 2.2).
    # link to fairseq implementation, search for nn.init to see them do the initialization
    # https://fairseq.readthedocs.io/en/v0.7.1/_modules/fairseq/models/transformer.html
    #
    # For PyTorch, PyTorch initializes nn.Embedding with a standard normal distribution mean 0, variance 1: N(0,1).
    # this puts the range for the word embeddings around (-3, 3). the pytorch implementation for positional encoding
    # also has a range of (-1.0, 1.0). So already, these are much closer in scale, and it doesn't seem like we need
    # to increase the scale of the word embeddings. However, PyTorch example still multiply by the scaling factor
    # unclear whether this is just a carryover that is not actually needed, or if there is a different reason
    #
    # EDIT! I just realized that even though nn.Embedding defaults to a range of around (-3, 3), the PyTorch
    # transformer example actually re-initializes them using a uniform distribution in the range of (-0.1, 0.1)
    # that makes it very similar to the fairseq implementation, so the scaling factor that PyTorch uses actually would
    # bring the word embedding and positional encodings much closer in scale. So this could be the reason why pytorch
    # does it

    # Regarding #3, I don't think so. Firstly, does it actually cancel there? Secondly, the purpose of the scaling
    # factor in scaled dot product attention, according to attention is all you need, is to counteract dot products
    # that are very high in magnitude due to choice of large mbedding length (aka d_k). The problem with high magnitude
    # dot products is that potentially, the softmax is pushed into regions where it has extremely small gradients,
    # making learning difficult. If the scaling factor in the embedding was meant to counteract the scaling factor in
    # scaled dot product attention, then what would be the point of doing all that?

    # Regarding #4, I don't think the scaling will have any effects in practice, it's probably not needed

    # Overall, I think #2 is the most likely reason why this scaling is performed. In theory, I think
    # even if the scaling wasn't performed, the network might learn to up-scale the word embedding weights to increase
    # word embedding signal vs. the position signal on its own. Another question I have is why not just initialize
    # the embedding weights to have higher initial values? Why put it in the range (-0.1, 0.1)?
    #
    # The fact that most implementations have this scaling concerns me, makes me think I might be missing something.
    # For our purposes, we can train a couple models to see if scaling has any positive or negative effect.
    # Still need to think about potential effects of this scaling on relative position embeddings.

    def __init__(self, num_embeddings: int, embedding_dim: int, scale: bool):
        super(ScaledEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.emb_size = embedding_dim
        self.embed_scale = math.sqrt(self.emb_size)

        self.scale = scale

        self.init_weights()

    def init_weights(self):
        # todo: not sure why PyTorch example initializes weights like this
        #   might have something to do with word embedding scaling factor (see above)
        #   could also just try the default weight initialization for nn.Embedding()
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)

    def forward(self, tokens: Tensor, **kwargs):
        if self.scale:
            return self.embedding(tokens.long()) * self.embed_scale
        else:
            return self.embedding(tokens.long())


class FCBlock(nn.Module):
    """ a fully connected block with options for batchnorm and dropout
        can extend in the future with option for different activation, etc """

    def __init__(self,
                 in_features: int,
                 num_hidden_nodes: int = 64,
                 use_batchnorm: bool = False,
                 use_layernorm: bool = False,
                 norm_before_activation: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.2,
                 activation: str = "relu"):

        super().__init__()

        if use_batchnorm and use_layernorm:
            raise ValueError("Only one of use_batchnorm or use_layernorm can be set to True")

        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.use_layernorm = use_layernorm
        self.norm_before_activation = norm_before_activation

        self.fc = nn.Linear(in_features=in_features, out_features=num_hidden_nodes)

        self.activation = get_activation_fn(activation, functional=False)

        if use_batchnorm:
            self.norm = nn.BatchNorm1d(num_hidden_nodes)

        if use_layernorm:
            self.norm = nn.LayerNorm(num_hidden_nodes)

        if use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, **kwargs):
        x = self.fc(x)

        # norm can be before or after activation, using flag
        if (self.use_batchnorm or self.use_layernorm) and self.norm_before_activation:
            x = self.norm(x)

        x = self.activation(x)

        # batchnorm being applied after activation, there is some discussion on this online
        if (self.use_batchnorm or self.use_layernorm) and not self.norm_before_activation:
            x = self.norm(x)

        # dropout being applied last
        if self.use_dropout:
            x = self.dropout(x)

        return x


class TaskSpecificPredictionLayers(nn.Module):
    """ Constructs num_tasks [dense(num_hidden_nodes)+relu+dense(1)] layers, each independently transforming input
        into a single output node. All num_tasks outputs are then concatenated into a single tensor. """

    # todo: the independent layers are run in sequence rather than in parallel, causing a slowdown that
    #   scales with the number of tasks. might be able to run in parallel by hacking convolution operation
    #   https://stackoverflow.com/questions/58374980/run-multiple-models-of-an-ensemble-in-parallel-with-pytorch
    #   https://github.com/pytorch/pytorch/issues/54147
    #   https://github.com/pytorch/pytorch/issues/36459

    def __init__(self,
                 num_tasks: int,
                 in_features: int,
                 num_hidden_nodes: int = 64,
                 use_batchnorm: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.2,
                 activation: str = "relu"):

        super().__init__()

        # each task-specific layer outputs a single node,
        # which can be combined with torch.cat into prediction vector
        self.task_specific_pred_layers = nn.ModuleList()
        for i in range(num_tasks):
            layers = [FCBlock(in_features=in_features,
                              num_hidden_nodes=num_hidden_nodes,
                              use_batchnorm=use_batchnorm,
                              use_dropout=use_dropout,
                              dropout_rate=dropout_rate,
                              activation=activation),
                      nn.Linear(in_features=num_hidden_nodes, out_features=1)]
            self.task_specific_pred_layers.append(nn.Sequential(*layers))

    def forward(self, x, **kwargs):
        # run each task-specific layer and concatenate outputs into a single output vector
        task_specific_outputs = []
        for layer in self.task_specific_pred_layers:
            task_specific_outputs.append(layer(x))

        output = torch.cat(task_specific_outputs, dim=1)
        return output


class GlobalAveragePooling(nn.Module):
    """ helper class for global average pooling """

    def __init__(self, dim=1):
        super().__init__()
        # our data is in [batch_size, sequence_length, embedding_length]
        # with global pooling, we want to pool over the sequence dimension (dim=1)
        self.dim = dim

    def forward(self, x, **kwargs):
        return torch.mean(x, dim=self.dim)


class CLSPooling(nn.Module):
    """ helper class for CLS token extraction """

    def __init__(self, cls_position=0):
        super().__init__()

        # the position of the CLS token in the sequence dimension
        # currently, the CLS token is in the first position, but may move it to the last position
        self.cls_position = cls_position

    def forward(self, x, **kwargs):
        # assumes input is in [batch_size, sequence_len, embedding_len]
        # thus sequence dimension is dimension 1
        return x[:, self.cls_position, :]


class TransformerEncoderWrapper(nn.TransformerEncoder):
    """ wrapper around PyTorch's TransformerEncoder that re-initializes layer parameters,
        so each transformer encoder layer has a different initialization """

    # todo: PyTorch is changing its transformer API... check up on and see if there is a better way
    def __init__(self, encoder_layer, num_layers, norm=None, reset_params=True):
        super().__init__(encoder_layer, num_layers, norm)
        if reset_params:
            self.apply(reset_parameters_helper)


class AttnModel(nn.Module):
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--pos_encoding', type=str, default="absolute",
                            choices=["none", "absolute", "relative", "relative_3D"],
                            help="what type of positional encoding to use")
        parser.add_argument('--pos_encoding_dropout', type=float, default=0.1,
                            help="out much dropout to use in positional encoding, for pos_encoding==absolute")
        parser.add_argument('--clipping_threshold', type=int, default=3,
                            help="clipping threshold for relative position embedding, for relative and relative_3D")
        parser.add_argument('--contact_threshold', type=int, default=7,
                            help="threshold, in angstroms, for contact map, for relative_3D")
        parser.add_argument('--embedding_len', type=int, default=128)
        parser.add_argument('--num_heads', type=int, default=2)
        parser.add_argument('--num_hidden', type=int, default=64)
        parser.add_argument('--num_enc_layers', type=int, default=2)
        parser.add_argument('--enc_layer_dropout', type=float, default=0.1)
        parser.add_argument('--use_final_encoder_norm', action="store_true", default=False)

        parser.add_argument('--global_average_pooling', action="store_true", default=False)
        parser.add_argument('--cls_pooling', action="store_true", default=False)

        parser.add_argument('--use_task_specific_layers', action="store_true", default=False,
                            help="exclusive with use_final_hidden_layer; takes priority over use_final_hidden_layer"
                                 " if both flags are set")
        parser.add_argument('--task_specific_hidden_nodes', type=int, default=64)
        parser.add_argument('--use_final_hidden_layer', action="store_true", default=False)
        parser.add_argument('--final_hidden_size', type=int, default=64)
        parser.add_argument('--use_final_hidden_layer_norm', action="store_true", default=False)
        parser.add_argument('--final_hidden_layer_norm_before_activation', action="store_true", default=False)
        parser.add_argument('--use_final_hidden_layer_dropout', action="store_true", default=False)
        parser.add_argument('--final_hidden_layer_dropout_rate', type=float, default=0.2)

        parser.add_argument('--activation', type=str, default="relu",
                            help="activation function used for all activations in the network")
        return parser

    def __init__(self,
                 # data args
                 num_tasks: int,
                 aa_seq_len: int,
                 num_tokens: int,
                 # transformer encoder model args
                 pos_encoding: str = "absolute",
                 pos_encoding_dropout: float = 0.1,
                 clipping_threshold: int = 3,
                 contact_threshold: int = 7,
                 pdb_fns: List[str] = None,
                 embedding_len: int = 64,
                 num_heads: int = 2,
                 num_hidden: int = 64,
                 num_enc_layers: int = 2,
                 enc_layer_dropout: float = 0.1,
                 use_final_encoder_norm: bool = False,
                 # pooling to fixed-length representation
                 global_average_pooling: bool = True,
                 cls_pooling: bool = False,
                 # prediction layers
                 use_task_specific_layers: bool = False,
                 task_specific_hidden_nodes: int = 64,
                 use_final_hidden_layer: bool = False,
                 final_hidden_size: int = 64,
                 use_final_hidden_layer_norm: bool = False,
                 final_hidden_layer_norm_before_activation: bool = False,
                 use_final_hidden_layer_dropout: bool = False,
                 final_hidden_layer_dropout_rate: float = 0.2,
                 # activation function
                 activation: str = "relu",
                 *args, **kwargs):

        super().__init__()

        # store embedding length for use in the forward function
        self.embedding_len = embedding_len
        self.aa_seq_len = aa_seq_len

        # build up layers
        layers = collections.OrderedDict()

        # amino acid embedding
        layers["embedder"] = ScaledEmbedding(num_embeddings=num_tokens, embedding_dim=embedding_len, scale=True)

        # absolute positional encoding
        if pos_encoding == "absolute":
            layers["pos_encoder"] = PositionalEncoding(embedding_len, dropout=pos_encoding_dropout, max_len=512)

        # transformer encoder layer for none or absolute positional encoding
        if pos_encoding in ["none", "absolute"]:
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_len,
                                                             nhead=num_heads,
                                                             dim_feedforward=num_hidden,
                                                             dropout=enc_layer_dropout,
                                                             activation=get_activation_fn(activation),
                                                             norm_first=True,
                                                             batch_first=True)

            # layer norm that is used after the transformer encoder layers
            # if the norm_first is False, this is *redundant* and not needed
            # but if norm_first is True, this can be used to normalize outputs from
            # the transformer encoder before inputting to the final fully connected layer
            encoder_norm = None
            if use_final_encoder_norm:
                encoder_norm = nn.LayerNorm(embedding_len)

            layers["tr_encoder"] = TransformerEncoderWrapper(encoder_layer=encoder_layer,
                                                             num_layers=num_enc_layers,
                                                             norm=encoder_norm)

        # transformer encoder layer for relative position encoding
        elif pos_encoding in ["relative", "relative_3D"]:
            relative_encoder_layer = RelativeTransformerEncoderLayer(d_model=embedding_len,
                                                                        nhead=num_heads,
                                                                        pos_encoding=pos_encoding,
                                                                        clipping_threshold=clipping_threshold,
                                                                        contact_threshold=contact_threshold,
                                                                        pdb_fns=pdb_fns,
                                                                        dim_feedforward=num_hidden,
                                                                        dropout=enc_layer_dropout,
                                                                        activation=get_activation_fn(activation),
                                                                        norm_first=True)

            encoder_norm = None
            if use_final_encoder_norm:
                encoder_norm = nn.LayerNorm(embedding_len)

            layers["tr_encoder"] = RelativeTransformerEncoder(encoder_layer=relative_encoder_layer,
                                                                 num_layers=num_enc_layers,
                                                                 norm=encoder_norm)

        # GLOBAL AVERAGE POOLING OR CLS TOKEN
        # set up the layers and output shapes (i.e. input shapes for the pred layer)
        if global_average_pooling:
            # pool over the sequence dimension
            layers["avg_pooling"] = GlobalAveragePooling(dim=1)
            pred_layer_input_features = embedding_len
        elif cls_pooling:
            layers["cls_pooling"] = CLSPooling(cls_position=0)
            pred_layer_input_features = embedding_len
        else:
            # no global average pooling or CLS token
            # sequence dimension is still there, just flattened
            layers["flatten"] = nn.Flatten()
            pred_layer_input_features = embedding_len * aa_seq_len

        # PREDICTION
        if use_task_specific_layers:
            # task specific prediction layers (nonlinear transform for each task)
            layers["prediction"] = TaskSpecificPredictionLayers(num_tasks=num_tasks,
                                                                in_features=pred_layer_input_features,
                                                                num_hidden_nodes=task_specific_hidden_nodes,
                                                                activation=activation)
        elif use_final_hidden_layer:
            # combined prediction linear (linear transform for each task)
            layers["fc1"] = FCBlock(in_features=pred_layer_input_features,
                                    num_hidden_nodes=final_hidden_size,
                                    use_batchnorm=False,
                                    use_layernorm=use_final_hidden_layer_norm,
                                    norm_before_activation=final_hidden_layer_norm_before_activation,
                                    use_dropout=use_final_hidden_layer_dropout,
                                    dropout_rate=final_hidden_layer_dropout_rate,
                                    activation=activation)

            layers["prediction"] = nn.Linear(in_features=final_hidden_size, out_features=num_tasks)
        else:
            layers["prediction"] = nn.Linear(in_features=pred_layer_input_features, out_features=num_tasks)

        # FINAL MODEL
        self.model = SequentialWithArgs(layers)

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)


class Transpose(nn.Module):
    """ helper layer to swap data from (batch, seq, channels) to (batch, channels, seq)
        used as a helper in the convolutional network which pytorch defaults to channels-first """

    def __init__(self, dims: Tuple[int, ...] = (1, 2)):
        super().__init__()
        self.dims = dims

    def forward(self, x, **kwargs):
        x = x.transpose(*self.dims).contiguous()
        return x


def conv1d_out_shape(seq_len, kernel_size, stride=1, pad=0, dilation=1):
    return (seq_len + (2 * pad) - (dilation * (kernel_size - 1)) - 1 // stride) + 1


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int = 1,
                 padding: str = "same",
                 use_batchnorm: bool = False,
                 use_layernorm: bool = False,
                 norm_before_activation: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.2,
                 activation: str = "relu"):

        super().__init__()

        if use_batchnorm and use_layernorm:
            raise ValueError("Only one of use_batchnorm or use_layernorm can be set to True")

        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm
        self.norm_before_activation = norm_before_activation
        self.use_dropout = use_dropout

        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              dilation=dilation)

        self.activation = get_activation_fn(activation, functional=False)

        if use_batchnorm:
            self.norm = nn.BatchNorm1d(out_channels)

        if use_layernorm:
            self.norm = nn.LayerNorm(out_channels)

        if use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, **kwargs):
        x = self.conv(x)

        # norm can be before or after activation, using flag
        if self.use_batchnorm and self.norm_before_activation:
            x = self.norm(x)
        elif self.use_layernorm and self.norm_before_activation:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        x = self.activation(x)

        # batchnorm being applied after activation, there is some discussion on this online
        if self.use_batchnorm and not self.norm_before_activation:
            x = self.norm(x)
        elif self.use_layernorm and not self.norm_before_activation:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        # dropout being applied after batchnorm, there is some discussion on this online
        if self.use_dropout:
            x = self.dropout(x)

        return x


class ConvModel2(nn.Module):
    """ convolutional source model that supports padded inputs, pooling, etc """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--use_embedding', action="store_true", default=False)
        parser.add_argument('--embedding_len', type=int, default=128)

        parser.add_argument('--num_conv_layers', type=int, default=1)
        parser.add_argument('--kernel_sizes', type=int, nargs="+", default=[7])
        parser.add_argument('--out_channels', type=int, nargs="+", default=[128])
        parser.add_argument('--dilations', type=int, nargs="+", default=[1])
        parser.add_argument('--padding', type=str, default="valid", choices=["valid", "same"])
        parser.add_argument('--use_conv_layer_norm', action="store_true", default=False)
        parser.add_argument('--conv_layer_norm_before_activation', action="store_true", default=False)
        parser.add_argument('--use_conv_layer_dropout', action="store_true", default=False)
        parser.add_argument('--conv_layer_dropout_rate', type=float, default=0.2)

        parser.add_argument('--global_average_pooling', action="store_true", default=False)

        parser.add_argument('--use_task_specific_layers', action="store_true", default=False)
        parser.add_argument('--task_specific_hidden_nodes', type=int, default=64)
        parser.add_argument('--use_final_hidden_layer', action="store_true", default=False)
        parser.add_argument('--final_hidden_size', type=int, default=64)
        parser.add_argument('--use_final_hidden_layer_norm', action="store_true", default=False)
        parser.add_argument('--final_hidden_layer_norm_before_activation', action="store_true", default=False)
        parser.add_argument('--use_final_hidden_layer_dropout', action="store_true", default=False)
        parser.add_argument('--final_hidden_layer_dropout_rate', type=float, default=0.2)

        parser.add_argument('--activation', type=str, default="relu",
                            help="activation function used for all activations in the network")

        return parser

    def __init__(self,
                 # data
                 num_tasks: int,
                 aa_seq_len: int,
                 aa_encoding_len: int,
                 num_tokens: int,
                 # convolutional model args
                 use_embedding: bool = False,
                 embedding_len: int = 64,
                 num_conv_layers: int = 1,
                 kernel_sizes: List[int] = (7,),
                 out_channels: List[int] = (128,),
                 dilations: List[int] = (1,),
                 padding: str = "valid",
                 use_conv_layer_norm: bool = False,
                 conv_layer_norm_before_activation: bool = False,
                 use_conv_layer_dropout: bool = False,
                 conv_layer_dropout_rate: float = 0.2,
                 # pooling
                 global_average_pooling: bool = True,
                 # prediction layers
                 use_task_specific_layers: bool = False,
                 task_specific_hidden_nodes: int = 64,
                 use_final_hidden_layer: bool = False,
                 final_hidden_size: int = 64,
                 use_final_hidden_layer_norm: bool = False,
                 final_hidden_layer_norm_before_activation: bool = False,
                 use_final_hidden_layer_dropout: bool = False,
                 final_hidden_layer_dropout_rate: float = 0.2,
                 # activation function
                 activation: str = "relu",
                 *args, **kwargs):

        super(ConvModel2, self).__init__()

        # build up the layers
        layers = collections.OrderedDict()

        # amino acid embedding
        if use_embedding:
            layers["embedder"] = ScaledEmbedding(num_embeddings=num_tokens, embedding_dim=embedding_len, scale=False)

        # transpose the input to match PyTorch's expected format
        layers["transpose"] = Transpose(dims=(1, 2))

        # build up the convolutional layers
        for layer_num in range(num_conv_layers):
            # determine the number of input channels for the first convolutional layer
            if layer_num == 0 and use_embedding:
                # for the first convolutional layer, the in_channels is the embedding_len
                in_channels = embedding_len
            elif layer_num == 0 and not use_embedding:
                # for the first convolutional layer, the in_channels is the aa_encoding_len
                in_channels = aa_encoding_len
            else:
                in_channels = out_channels[layer_num - 1]

            layers[f"conv{layer_num}"] = ConvBlock(in_channels=in_channels,
                                                   out_channels=out_channels[layer_num],
                                                   kernel_size=kernel_sizes[layer_num],
                                                   dilation=dilations[layer_num],
                                                   padding=padding,
                                                   use_batchnorm=False,
                                                   use_layernorm=use_conv_layer_norm,
                                                   norm_before_activation=conv_layer_norm_before_activation,
                                                   use_dropout=use_conv_layer_dropout,
                                                   dropout_rate=conv_layer_dropout_rate,
                                                   activation=activation)

        # handle transition from convolutional layers to fully connected layer
        # either use global average pooling or flatten
        # take into consideration whether we are using valid or same padding
        if global_average_pooling:
            # global average pooling (mean across the seq len dimension)
            # the seq len dimensions is the last dimension (batch_size, num_filters, seq_len)
            layers["avg_pooling"] = GlobalAveragePooling(dim=-1)
            # the prediction layers will take num_filters input features
            pred_layer_input_features = out_channels[-1]

        else:
            # no global average pooling. flatten instead.
            layers["flatten"] = nn.Flatten()
            # calculate the final output len of the convolutional layers
            # and the number of input features for the prediction layers
            if padding == "valid":
                # valid padding (aka no padding) results in shrinking length in progressive layers
                conv_out_len = conv1d_out_shape(aa_seq_len, kernel_size=kernel_sizes[0], dilation=dilations[0])
                for layer_num in range(1, num_conv_layers):
                    conv_out_len = conv1d_out_shape(conv_out_len,
                                                    kernel_size=kernel_sizes[layer_num],
                                                    dilation=dilations[layer_num])
                pred_layer_input_features = conv_out_len * out_channels[-1]
            else:
                # padding == "same"
                pred_layer_input_features = aa_seq_len * out_channels[-1]

        # prediction layer
        if use_task_specific_layers:
            layers["prediction"] = TaskSpecificPredictionLayers(num_tasks=num_tasks,
                                                                in_features=pred_layer_input_features,
                                                                num_hidden_nodes=task_specific_hidden_nodes,
                                                                activation=activation)

        # final hidden layer (with potential additional dropout)
        elif use_final_hidden_layer:
            layers["fc1"] = FCBlock(in_features=pred_layer_input_features,
                                    num_hidden_nodes=final_hidden_size,
                                    use_batchnorm=False,
                                    use_layernorm=use_final_hidden_layer_norm,
                                    norm_before_activation=final_hidden_layer_norm_before_activation,
                                    use_dropout=use_final_hidden_layer_dropout,
                                    dropout_rate=final_hidden_layer_dropout_rate,
                                    activation=activation)
            layers["prediction"] = nn.Linear(in_features=final_hidden_size, out_features=num_tasks)

        else:
            layers["prediction"] = nn.Linear(in_features=pred_layer_input_features, out_features=num_tasks)

        self.model = nn.Sequential(layers)

    def forward(self, x, **kwargs):
        output = self.model(x)
        return output


class ConvModel(nn.Module):
    """ a convolutional network with convolutional layers followed by a fully connected layer """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_conv_layers', type=int, default=1)
        parser.add_argument('--kernel_sizes', type=int, nargs="+", default=[7])
        parser.add_argument('--out_channels', type=int, nargs="+", default=[128])
        parser.add_argument('--padding', type=str, default="valid", choices=["valid", "same"])
        parser.add_argument('--use_final_hidden_layer', action="store_true",
                            help="whether to use a final hidden layer")
        parser.add_argument('--final_hidden_size', type=int, default=128,
                            help="number of nodes in the final hidden layer")
        parser.add_argument('--use_dropout', action="store_true",
                            help="whether to use dropout in the final hidden layer")
        parser.add_argument('--dropout_rate', type=float, default=0.2,
                            help="dropout rate in the final hidden layer")
        parser.add_argument('--use_task_specific_layers', action="store_true", default=False)
        parser.add_argument('--task_specific_hidden_nodes', type=int, default=64)
        return parser

    def __init__(self,
                 num_tasks: int,
                 aa_seq_len: int,
                 aa_encoding_len: int,
                 num_conv_layers: int = 1,
                 kernel_sizes: List[int] = (7,),
                 out_channels: List[int] = (128,),
                 padding: str = "valid",
                 use_final_hidden_layer: bool = True,
                 final_hidden_size: int = 128,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.2,
                 use_task_specific_layers: bool = False,
                 task_specific_hidden_nodes: int = 64,
                 *args, **kwargs):

        super(ConvModel, self).__init__()

        # set up the model as a Sequential block (less to do in forward())
        layers = collections.OrderedDict()

        layers["transpose"] = Transpose(dims=(1, 2))

        for layer_num in range(num_conv_layers):
            # for the first convolutional layer, the in_channels is the feature_len
            in_channels = aa_encoding_len if layer_num == 0 else out_channels[layer_num - 1]

            layers["conv{}".format(layer_num)] = nn.Sequential(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels[layer_num],
                          kernel_size=kernel_sizes[layer_num],
                          padding=padding),
                nn.ReLU()
            )

        layers["flatten"] = nn.Flatten()

        # calculate the final output len of the convolutional layers
        # and the number of input features for the prediction layers
        if padding == "valid":
            # valid padding (aka no padding) results in shrinking length in progressive layers
            conv_out_len = conv1d_out_shape(aa_seq_len, kernel_size=kernel_sizes[0])
            for layer_num in range(1, num_conv_layers):
                conv_out_len = conv1d_out_shape(conv_out_len, kernel_size=kernel_sizes[layer_num])
            next_dim = conv_out_len * out_channels[-1]
        elif padding == "same":
            next_dim = aa_seq_len * out_channels[-1]
        else:
            raise ValueError("unexpected value for padding: {}".format(padding))

        # final hidden layer (with potential additional dropout)
        if use_final_hidden_layer:
            layers["fc1"] = FCBlock(in_features=next_dim,
                                    num_hidden_nodes=final_hidden_size,
                                    use_batchnorm=False,
                                    use_dropout=use_dropout,
                                    dropout_rate=dropout_rate)
            next_dim = final_hidden_size

        # final prediction layer
        # either task specific nonlinear layers or a single linear layer
        if use_task_specific_layers:
            layers["prediction"] = TaskSpecificPredictionLayers(num_tasks=num_tasks,
                                                                in_features=next_dim,
                                                                num_hidden_nodes=task_specific_hidden_nodes)
        else:
            layers["prediction"] = nn.Linear(in_features=next_dim, out_features=num_tasks)

        self.model = nn.Sequential(layers)

    def forward(self, x, **kwargs):
        output = self.model(x)
        return output


class FCModel(nn.Module):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--num_hidden', nargs="+", type=int, default=[128])
        parser.add_argument('--use_batchnorm', action="store_true", default=False)
        parser.add_argument('--use_layernorm', action="store_true", default=False)
        parser.add_argument('--norm_before_activation', action="store_true", default=False)
        parser.add_argument('--use_dropout', action="store_true", default=False)
        parser.add_argument('--dropout_rate', type=float, default=0.2)
        return parser

    def __init__(self,
                 num_tasks: int,
                 seq_encoding_len: int,
                 num_layers: int = 1,
                 num_hidden: List[int] = (128,),
                 use_batchnorm: bool = False,
                 use_layernorm: bool = False,
                 norm_before_activation: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.2,
                 activation: str = "relu",
                 *args, **kwargs):
        super().__init__()

        # set up the model as a Sequential block (less to do in forward())
        layers = collections.OrderedDict()

        # flatten inputs as this is all fully connected
        layers["flatten"] = nn.Flatten()

        # build up the variable number of hidden layers (fully connected + ReLU + dropout (if set))
        for layer_num in range(num_layers):
            # for the first layer (layer_num == 0), in_features is determined by given input
            # for subsequent layers, the in_features is the previous layer's num_hidden
            in_features = seq_encoding_len if layer_num == 0 else num_hidden[layer_num - 1]

            layers["fc{}".format(layer_num)] = FCBlock(in_features=in_features,
                                                       num_hidden_nodes=num_hidden[layer_num],
                                                       use_batchnorm=use_batchnorm,
                                                       use_layernorm=use_layernorm,
                                                       norm_before_activation=norm_before_activation,
                                                       use_dropout=use_dropout,
                                                       dropout_rate=dropout_rate,
                                                       activation=activation)

        # finally, the linear output layer
        in_features = num_hidden[-1] if num_layers > 0 else seq_encoding_len
        layers["output"] = nn.Linear(in_features=in_features, out_features=num_tasks)

        self.model = nn.Sequential(layers)

    def forward(self, x, **kwargs):
        output = self.model(x)
        return output


class LRModel(nn.Module):
    """ a simple linear model """

    def __init__(self, num_tasks, seq_encoding_len, *args, **kwargs):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_encoding_len, out_features=num_tasks))

    def forward(self, x, **kwargs):
        output = self.model(x)
        return output


class TransferModel(nn.Module):
    """ transfer learning model """

    @staticmethod
    def add_model_specific_args(parent_parser):

        def none_or_int(value: str):
            return None if value.lower() == "none" else int(value)

        p = ArgumentParser(parents=[parent_parser], add_help=False)

        # for model set up
        p.add_argument('--pretrained_ckpt_path', type=str, default=None)

        # where to cut off the backbone
        p.add_argument("--backbone_cutoff", type=none_or_int, default=-1,
                       help="where to cut off the backbone. can be a negative int, indexing back from "
                            "pretrained_model.model.model. a value of -1 would chop off the backbone prediction head. "
                            "a value of -2 chops the prediction head and FC layer. a value of -3 chops"
                            "the above, as well as the global average pooling layer. all depends on architecture.")

        p.add_argument("--pred_layer_input_features", type=int, default=None,
                       help="if None, number of features will be determined based on backbone_cutoff and standard "
                            "architecture. otherwise, specify the number of input features for the prediction layer")

        # top net args
        p.add_argument("--top_net_type", type=str, default="linear", choices=["linear", "nonlinear", "sklearn"])
        p.add_argument("--top_net_hidden_nodes", type=int, default=256)
        p.add_argument("--top_net_use_batchnorm", action="store_true")
        p.add_argument("--top_net_use_dropout", action="store_true")
        p.add_argument("--top_net_dropout_rate", type=float, default=0.1)

        return p

    def __init__(self,
                 # pretrained model
                 pretrained_ckpt_path: Optional[str] = None,
                 pretrained_hparams: Optional[dict] = None,
                 backbone_cutoff: Optional[int] = -1,
                 # top net
                 pred_layer_input_features: Optional[int] = None,
                 top_net_type: str = "linear",
                 top_net_hidden_nodes: int = 256,
                 top_net_use_batchnorm: bool = False,
                 top_net_use_dropout: bool = False,
                 top_net_dropout_rate: float = 0.1,
                 *args, **kwargs):

        super().__init__()

        # error checking: if pretrained_ckpt_path is None, then pretrained_hparams must be specified
        if pretrained_ckpt_path is None and pretrained_hparams is None:
            raise ValueError("Either pretrained_ckpt_path or pretrained_hparams must be specified")

        # note: pdb_fns is loaded from transfer model arguments rather than original source model hparams
        # if pdb_fns is specified as a kwarg, pass it on for structure-based RPE
        # otherwise, can just set pdb_fns to None, and structure-based RPE will handle new PDBs on the fly
        pdb_fns = kwargs["pdb_fns"] if "pdb_fns" in kwargs else None

        # generate a fresh backbone using pretrained_hparams if specified
        # otherwise load the backbone from the pretrained checkpoint
        # we prioritize pretrained_hparams over pretrained_ckpt_path because
        # pretrained_hparams will only really be specified if we are loading from a DMSTask checkpoint
        # meaning the TransferModel has already been fine-tuned on DMS data, and we are likely loading
        # weights from that finetuning (including weights for the backbone)
        # whereas if pretrained_hparams is not specified but pretrained_ckpt_path is, then we are
        # likely finetuning the TransferModel for the first time, and we need the pretrained weights for the
        # backbone from the RosettaTask checkpoint
        if pretrained_hparams is not None:
            # pretrained_hparams will only be specified if we are loading from a DMSTask checkpoint
            pretrained_hparams["pdb_fns"] = pdb_fns
            pretrained_model = Model[pretrained_hparams["model_name"]].cls(**pretrained_hparams)
            self.pretrained_hparams = pretrained_hparams
        else:
            # not supported in metl-pretrained
            raise NotImplementedError("Loading pretrained weights from RosettaTask checkpoint not supported")

        layers = collections.OrderedDict()

        # set the backbone to all layers except the last layer (the pre-trained prediction layer)
        if backbone_cutoff is None:
            layers["backbone"] = SequentialWithArgs(*list(pretrained_model.model.children()))
        else:
            layers["backbone"] = SequentialWithArgs(*list(pretrained_model.model.children())[0:backbone_cutoff])

        if top_net_type == "sklearn":
            # sklearn top not doesn't require any more layers, just return model for the repr layer
            self.model = SequentialWithArgs(layers)
            return

        # figure out dimensions of input into the prediction layer
        if pred_layer_input_features is None:
            # todo: can make this more robust by checking if the pretrained_mode.hparams for use_final_hidden_layer,
            #   global_average_pooling, etc. then can determine what the layer will be based on backbone_cutoff.
            # currently, assumes that pretrained_model uses global average pooling and a final_hidden_layer
            if backbone_cutoff is None:
                # no backbone cutoff... use the full network (including tasks) as the backbone
                pred_layer_input_features = self.pretrained_hparams["num_tasks"]
            elif backbone_cutoff == -1:
                pred_layer_input_features = self.pretrained_hparams["final_hidden_size"]
            elif backbone_cutoff == -2:
                pred_layer_input_features = self.pretrained_hparams["embedding_len"]
            elif backbone_cutoff == -3:
                pred_layer_input_features = self.pretrained_hparams["embedding_len"] * kwargs["aa_seq_len"]
            else:
                raise ValueError("can't automatically determine pred_layer_input_features for given backbone_cutoff")

        layers["flatten"] = nn.Flatten(start_dim=1)

        # create a new prediction layer on top of the backbone
        if top_net_type == "linear":
            # linear layer for prediction
            layers["prediction"] = nn.Linear(in_features=pred_layer_input_features, out_features=1)
        elif top_net_type == "nonlinear":
            # fully connected with hidden layer
            fc_block = FCBlock(in_features=pred_layer_input_features,
                               num_hidden_nodes=top_net_hidden_nodes,
                               use_batchnorm=top_net_use_batchnorm,
                               use_dropout=top_net_use_dropout,
                               dropout_rate=top_net_dropout_rate)

            pred_layer = nn.Linear(in_features=top_net_hidden_nodes, out_features=1)

            layers["prediction"] = SequentialWithArgs(fc_block, pred_layer)
        else:
            raise ValueError("Unexpected type of top net layer: {}".format(top_net_type))

        self.model = SequentialWithArgs(layers)

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)


def get_activation_fn(activation, functional=True):
    if activation == "relu":
        return F.relu if functional else nn.ReLU()
    elif activation == "gelu":
        return F.gelu if functional else nn.GELU()
    elif activation == "silo" or activation == "swish":
        return F.silu if functional else nn.SiLU()
    elif activation == "leaky_relu" or activation == "lrelu":
        return F.leaky_relu if functional else nn.LeakyReLU()
    else:
        raise RuntimeError("unknown activation: {}".format(activation))


class Model(enum.Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, cls, transfer_model):
        self.cls = cls
        self.transfer_model = transfer_model

    linear = LRModel, False
    fully_connected = FCModel, False
    cnn = ConvModel, False
    cnn2 = ConvModel2, False
    transformer_encoder = AttnModel, False
    transfer_model = TransferModel, True


def main():
    pass


if __name__ == "__main__":
    main()

## Relative attention

""" implementation of transformer encoder with relative attention
    references:
        - https://medium.com/@_init_/how-self-attention-with-relative-position-representations-works-28173b8c245a
        - https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
        - https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
        - https://github.com/jiezouguihuafu/ClassicalModelreproduced/blob/main/Transformer/transfor_rpe.py
"""

# import copy
# from os.path import basename, dirname, join, isfile
# from typing import Optional, Union

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# from torch.nn import Linear, Dropout, LayerNorm
# import time
# import networkx as nx

# import metl.structure as structure
# import metl.models as models


class RelativePosition3D(nn.Module):
    """ Contact map-based relative position embeddings """

    #  need to compute a bucket_mtx for each structure
    #  need to know which bucket_mtx to use when grabbing the embeddings in forward()
    #   - on init, get a list of all PDB files we will be using
    #   - use a dictionary to store PDB files --> bucket_mtxs
    #   - forward() gets a new arg: the pdb file, which indexes into the dictionary to grab the right bucket_mtx
    def __init__(self,
                 embedding_len: int,
                 contact_threshold: int,
                 clipping_threshold: int,
                 pdb_fns: Optional[Union[str, list, tuple]] = None,
                 default_pdb_dir: str = "data/pdb_files"):

        # preferably, pdb_fns contains full paths to the PDBs, but if just the PDB filename is given
        # then it defaults to the path data/pdb_files/<pdb_fn>
        super().__init__()
        self.embedding_len = embedding_len
        self.clipping_threshold = clipping_threshold
        self.contact_threshold = contact_threshold
        self.default_pdb_dir = default_pdb_dir

        # dummy buffer for getting correct device for on-the-fly bucket matrix generation
        self.register_buffer("dummy_buffer", torch.empty(0), persistent=False)

        # for 3D-based positions, the number of embeddings is generally the number of buckets
        # for contact map-based distances, that is clipping_threshold + 1
        num_embeddings = clipping_threshold + 1

        # this is the embedding lookup table E_r
        self.embeddings_table = nn.Embedding(num_embeddings, embedding_len)

        # set up pdb_fns that were passed in on init (can also be set up during runtime in forward())
        # todo: i'm using a hacky workaround to move the bucket_mtxs to the correct device
        #   i tried to make it more efficient by registering bucket matrices as buffers, but i was
        #   having problems with DDP syncing the buffers across processes
        self.bucket_mtxs = {}
        self.bucket_mtxs_device = self.dummy_buffer.device
        self._init_pdbs(pdb_fns)

    def forward(self, pdb_fn):
        # compute matrix R by grabbing the embeddings from the embeddings lookup table
        embeddings = self.embeddings_table(self._get_bucket_mtx(pdb_fn))
        return embeddings

    # def _get_bucket_mtx(self, pdb_fn):
    #     """ retrieve a bucket matrix given the pdb_fn.
    #         if the pdb_fn was provided at init or has already been computed, then the bucket matrix will be
    #         retrieved from the object buffer. if the bucket matrix has not been computed yet, it will be here """
    #     pdb_attr = self._pdb_key(pdb_fn)
    #     if hasattr(self, pdb_attr):
    #         return getattr(self, pdb_attr)
    #     else:
    #         # encountering a new PDB at runtime... process it
    #         # todo: if there's a new PDB at runtime, it will be initialized separately in each instance
    #         #   of RelativePosition3D, for each layer. It would be more efficient to have a global
    #         #   bucket_mtx registry... perhaps in the RelativeTransformerEncoder class, that can be passed through
    #         self._init_pdb(pdb_fn)
    #         return getattr(self, pdb_attr)

    def _move_bucket_mtxs(self, device):
        for k, v in self.bucket_mtxs.items():
            self.bucket_mtxs[k] = v.to(device)
        self.bucket_mtxs_device = device

    def _get_bucket_mtx(self, pdb_fn):
        """ retrieve a bucket matrix given the pdb_fn.
            if the pdb_fn was provided at init or has already been computed, then the bucket matrix will be
            retrieved from the bucket_mtxs dictionary. else, it will be computed now on-the-fly """

        # ensure that all the bucket matrices are on the same device as the nn.Embedding
        if self.bucket_mtxs_device != self.dummy_buffer.device:
            self._move_bucket_mtxs(self.dummy_buffer.device)

        pdb_attr = self._pdb_key(pdb_fn)
        if pdb_attr in self.bucket_mtxs:
            return self.bucket_mtxs[pdb_attr]
        else:
            # encountering a new PDB at runtime... process it
            # todo: if there's a new PDB at runtime, it will be initialized separately in each instance
            #   of RelativePosition3D, for each layer. It would be more efficient to have a global
            #   bucket_mtx registry... perhaps in the RelativeTransformerEncoder class, that can be passed through
            self._init_pdb(pdb_fn)
            return self.bucket_mtxs[pdb_attr]

    # def _set_bucket_mtx(self, pdb_fn, bucket_mtx):
    #     """ store a bucket matrix as a buffer """
    #     # if PyTorch ever implements a BufferDict, we could use it here efficiently
    #     # there is also BufferDict from https://botorch.org/api/_modules/botorch/utils/torch.html
    #     # would just need to modify it to have an option for persistent=False
    #     bucket_mtx = bucket_mtx.to(self.dummy_buffer.device)
    #
    #     self.register_buffer(self._pdb_key(pdb_fn), bucket_mtx, persistent=False)

    def _set_bucket_mtx(self, pdb_fn, bucket_mtx):
        """ store a bucket matrix in the bucket dict """

        # move the bucket_mtx to the same device that the other bucket matrices are on
        bucket_mtx = bucket_mtx.to(self.bucket_mtxs_device)

        self.bucket_mtxs[self._pdb_key(pdb_fn)] = bucket_mtx

    @staticmethod
    def _pdb_key(pdb_fn):
        """ return a unique key for the given pdb_fn, used to map unique PDBs """
        # note this key does NOT currently support PDBs with the same basename but different paths
        # assumes every PDB is in the format <pdb_name>.pdb
        # should be a compatible with being a class attribute, as it is used as a pytorch buffer name
        return f"pdb_{basename(pdb_fn).split('.')[0]}"

    def _init_pdbs(self, pdb_fns):
        start = time.time()

        if pdb_fns is None:
            # nothing to initialize if pdb_fns is None
            return

        # make sure pdb_fns is a list
        if not isinstance(pdb_fns, list) and not isinstance(pdb_fns, tuple):
            pdb_fns = [pdb_fns]

        # init each pdb fn in the list
        for pdb_fn in pdb_fns:
            self._init_pdb(pdb_fn)

        print("Initialized PDB bucket matrices in: {:.3f}".format(time.time() - start))

    def _init_pdb(self, pdb_fn):
        """ process a pdb file for use with structure-based relative attention """
        # if pdb_fn is not a full path, default to the path data/pdb_files/<pdb_fn>
        if dirname(pdb_fn) == "":
            # handle the case where the pdb file is in the current working directory
            # if there is a PDB file in the cwd.... then just use it as is. otherwise, append the default.
            if not isfile(pdb_fn):
                pdb_fn = join(self.default_pdb_dir, pdb_fn)

        # create a structure graph from the pdb_fn and contact threshold
        cbeta_mtx = cbeta_distance_matrix(pdb_fn)
        structure_graph = dist_thresh_graph(cbeta_mtx, self.contact_threshold)

        # bucket_mtx indexes into the embedding lookup table to create the final distance matrix
        bucket_mtx = self._compute_bucket_mtx(structure_graph)

        self._set_bucket_mtx(pdb_fn, bucket_mtx)

    def _compute_bucketed_neighbors(self, structure_graph, source_node):
        """ gets the bucketed neighbors from the given source node and structure graph"""
        if self.clipping_threshold < 0:
            raise ValueError("Clipping threshold must be >= 0")

        sspl = _inv_dict(nx.single_source_shortest_path_length(structure_graph, source_node))

        if self.clipping_threshold is not None:
            num_buckets = 1 + self.clipping_threshold
            sspl = _combine_d(sspl, self.clipping_threshold, num_buckets - 1)

        return sspl

    def _compute_bucket_mtx(self, structure_graph):
        """ get the bucket_mtx for the given structure_graph
            calls _get_bucketed_neighbors for every node in the structure_graph """
        num_residues = len(list(structure_graph))

        # index into the embedding lookup table to create the final distance matrix
        bucket_mtx = torch.zeros(num_residues, num_residues, dtype=torch.long)

        for node_num in sorted(list(structure_graph)):
            bucketed_neighbors = self._compute_bucketed_neighbors(structure_graph, node_num)

            for bucket_num, neighbors in bucketed_neighbors.items():
                bucket_mtx[node_num, neighbors] = bucket_num

        return bucket_mtx


class RelativePosition(nn.Module):
    """ creates the embedding lookup table E_r and computes R
        note this inherits from pl.LightningModule instead of nn.Module
        makes it easier to access the device with `self.device`
        might be able to keep it as an nn.Module using the hacky dummy_param or commented out .device property """

    def __init__(self, embedding_len: int, clipping_threshold: int):
        """
        embedding_len: the length of the embedding, may be d_model, or d_model // num_heads for multihead
        clipping_threshold: the maximum relative position, referred to as k by Shaw et al.
        """
        super().__init__()
        self.embedding_len = embedding_len
        self.clipping_threshold = clipping_threshold
        # for sequence-based distances, the number of embeddings is 2*k+1, where k is the clipping threshold
        num_embeddings = 2 * clipping_threshold + 1

        # this is the embedding lookup table E_r
        self.embeddings_table = nn.Embedding(num_embeddings, embedding_len)

        # for getting the correct device for range vectors in forward
        self.register_buffer("dummy_buffer", torch.empty(0), persistent=False)

    def forward(self, length_q, length_k):
        # supports different length sequences, but in self-attention length_q and length_k are the same
        range_vec_q = torch.arange(length_q, device=self.dummy_buffer.device)
        range_vec_k = torch.arange(length_k, device=self.dummy_buffer.device)

        # this sets up the standard sequence-based distance matrix for relative positions
        # the current position is 0, positions to the right are +1, +2, etc, and to the left -1, -2, etc
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.clipping_threshold, self.clipping_threshold)

        # convert to indices, indexing into the embedding table
        final_mat = (distance_mat_clipped + self.clipping_threshold).long()

        # compute matrix R by grabbing the embeddings from the embedding lookup table
        embeddings = self.embeddings_table(final_mat)

        return embeddings


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, pos_encoding, clipping_threshold, contact_threshold, pdb_fns):
        """
        Multi-head attention with relative position embeddings.  Input data should be in batch_first format.
        :param embed_dim: aka d_model, aka hid_dim
        :param num_heads: number of heads
        :param dropout: how much dropout for scaled dot product attention

        :param pos_encoding: what type of positional encoding to use, relative or relative3D
        :param clipping_threshold: clipping threshold for relative position embedding
        :param contact_threshold: for relative_3D, the threshold in angstroms for the contact map
        :param pdb_fns: pdb file(s) to set up the relative position object

        """
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # model dimensions
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # pos encoding stuff
        self.pos_encoding = pos_encoding
        self.clipping_threshold = clipping_threshold
        self.contact_threshold = contact_threshold
        if pdb_fns is not None and not isinstance(pdb_fns, list):
            pdb_fns = [pdb_fns]
        self.pdb_fns = pdb_fns

        # relative position embeddings for use with keys and values
        # Shaw et al. uses relative position information for both keys and values
        # Huang et al. only uses it for the keys, which is probably enough
        if pos_encoding == "relative":
            self.relative_position_k = RelativePosition(self.head_dim, self.clipping_threshold)
            self.relative_position_v = RelativePosition(self.head_dim, self.clipping_threshold)
        elif pos_encoding == "relative_3D":
            self.relative_position_k = RelativePosition3D(self.head_dim, self.contact_threshold,
                                                          self.clipping_threshold, self.pdb_fns)
            self.relative_position_v = RelativePosition3D(self.head_dim, self.contact_threshold,
                                                          self.clipping_threshold, self.pdb_fns)
        else:
            raise ValueError("unrecognized pos_encoding: {}".format(pos_encoding))

        # WQ, WK, and WV from attention is all you need
        # note these default to bias=True, same as PyTorch implementation
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # WO from attention is all you need
        # used for the final projection when computing multi-head attention
        # PyTorch uses NonDynamicallyQuantizableLinear instead of Linear to avoid triggering an obscure
        # error quantizing the model https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L122
        # todo: if quantizing the model, explore if the above is a concern for us
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # dropout for scaled dot product attention
        self.dropout = nn.Dropout(dropout)

        # scaling factor for scaled dot product attention
        scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        # persistent=False if you don't want to save it inside state_dict
        self.register_buffer('scale', scale)

        # toggles meant to be set directly by user
        self.need_weights = False
        self.average_attn_weights = True

    def _compute_attn_weights(self, query, key, len_q, len_k, batch_size, mask, pdb_fn):
        """ computes the attention weights (a "compatability function" of queries with corresponding keys) """

        # calculate the first term in the numerator attn1, which is Q*K
        # todo: pytorch reshapes q,k and v to 3 dimensions (similar to how r_q2 is below)
        #   is that functionally equivalent to what we're doing? is their way faster?
        # r_q1 = [batch_size, num_heads, len_q, head_dim]
        r_q1 = query.view(batch_size, len_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # todo: we could directly permute r_k1 to [batch_size, num_heads, head_dim, len_k]
        #   to make it compatible for matrix multiplication with r_q1, instead of 2-step approach
        # r_k1 = [batch_size, num_heads, len_k, head_dim]
        r_k1 = key.view(batch_size, len_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # attn1 = [batch_size, num_heads, len_q, len_k]
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        # calculate the second term in the numerator attn2, which is Q*R
        # r_q2 = [query_len, batch_size * num_heads, head_dim]
        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size * self.num_heads, self.head_dim)

        # todo: support multiple different PDB base structures per batch
        #   one option:
        #       - require batches to be all the same protein
        #       - add argument to forward() to accept the PDB file for the protein in the batch
        #       - then we just pass in the PDB file to relative position's forward()
        #   to support multiple different structures per batch:
        #       - add argument to forward() to accept PDB files, one for each item in batch
        #       - make corresponding changing in relative_position object to return R for each structure
        #       - note: if there are a lot of of different structures, and the sequence lengths are long,
        #               this could be memory prohibitive because R (rel_pos_k) can take up a lot of mem for long seqs
        #       - adjust the attn2 calculation to factor in the multiple different R matrices.
        #               the way to do this might have to be to do multiple matmuls, one for each each structure.
        #               basically, would split up r_q2 into several matrices grouped by structure, and then
        #               multiply with corresponding R, then combine back into the exact same order of the original r_q2
        #               note: this may be computationally intensive (splitting, more matrix muliplies, joining)
        #               another option would be to create views(?), repeating the different Rs so we can do a
        #               a matris multiply directly with r_q2
        #       - would shapes be affected if there was padding in the queries, keys, values?

        if self.pos_encoding == "relative":
            # rel_pos_k = [len_q, len_k, head_dim]
            rel_pos_k = self.relative_position_k(len_q, len_k)
        elif self.pos_encoding == "relative_3D":
            # rel_pos_k = [sequence length (from PDB structure), head_dim]
            rel_pos_k = self.relative_position_k(pdb_fn)
        else:
            raise ValueError("unrecognized pos_encoding: {}".format(self.pos_encoding))

        # the matmul basically computes the dot product between each input position’s query vector and
        # its corresponding relative position embeddings across all input sequences in the heads and batch
        # attn2 = [batch_size * num_heads, len_q, len_k]
        attn2 = torch.matmul(r_q2, rel_pos_k.transpose(1, 2)).transpose(0, 1)
        # attn2 = [batch_size, num_heads, len_q, len_k]
        attn2 = attn2.contiguous().view(batch_size, self.num_heads, len_q, len_k)

        # calculate attention weights
        attn_weights = (attn1 + attn2) / self.scale

        # apply mask if given
        if mask is not None:
            # todo: pytorch uses float("-inf") instead of -1e10
            attn_weights = attn_weights.masked_fill(mask == 0, -1e10)

        # softmax gives us attn_weights weights
        attn_weights = torch.softmax(attn_weights, dim=-1)
        # attn_weights = [batch_size, num_heads, len_q, len_k]
        attn_weights = self.dropout(attn_weights)

        return attn_weights

    def _compute_avg_val(self, value, len_q, len_k, len_v, attn_weights, batch_size, pdb_fn):
        # todo: add option to not factor in relative position embeddings in value calculation
        # calculate the first term, the attn*values
        # r_v1 = [batch_size, num_heads, len_v, head_dim]
        r_v1 = value.view(batch_size, len_v, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # avg1 = [batch_size, num_heads, len_q, head_dim]
        avg1 = torch.matmul(attn_weights, r_v1)

        # calculate the second term, the attn*R
        # similar to how relative embeddings are factored in the attention weights calculation
        if self.pos_encoding == "relative":
            # rel_pos_v = [query_len, value_len, head_dim]
            rel_pos_v = self.relative_position_v(len_q, len_v)
        elif self.pos_encoding == "relative_3D":
            # rel_pos_v = [sequence length (from PDB structure), head_dim]
            rel_pos_v = self.relative_position_v(pdb_fn)
        else:
            raise ValueError("unrecognized pos_encoding: {}".format(self.pos_encoding))

        # r_attn_weights = [len_q, batch_size * num_heads, len_v]
        r_attn_weights = attn_weights.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size * self.num_heads, len_k)
        avg2 = torch.matmul(r_attn_weights, rel_pos_v)
        # avg2 = [batch_size, num_heads, len_q, head_dim]
        avg2 = avg2.transpose(0, 1).contiguous().view(batch_size, self.num_heads, len_q, self.head_dim)

        # calculate avg value
        x = avg1 + avg2  # [batch_size, num_heads, len_q, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, len_q, num_heads, head_dim]
        # x = [batch_size, len_q, embed_dim]
        x = x.view(batch_size, len_q, self.embed_dim)

        return x

    def forward(self, query, key, value, pdb_fn=None, mask=None):
        # query = [batch_size, q_len, embed_dim]
        # key = [batch_size, k_len, embed_dim]
        # value = [batch_size, v_en, embed_dim]
        batch_size = query.shape[0]
        len_k, len_q, len_v = (key.shape[1], query.shape[1], value.shape[1])

        # in projection (multiply inputs by WQ, WK, WV)
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # first compute the attention weights, then multiply with values
        # attn = [batch size, num_heads, len_q, len_k]
        attn_weights = self._compute_attn_weights(query, key, len_q, len_k, batch_size, mask, pdb_fn)

        # take weighted average of values (weighted by attention weights)
        attn_output = self._compute_avg_val(value, len_q, len_k, len_v, attn_weights, batch_size, pdb_fn)

        # output projection
        # attn_output = [batch_size, len_q, embed_dim]
        attn_output = self.out_proj(attn_output)

        if self.need_weights:
            # return attention weights in addition to attention
            # average the weights over the heads (to get overall attention)
            # attn_weights = [batch_size, len_q, len_k]
            if self.average_attn_weights:
                attn_weights = attn_weights.sum(dim=1) / self.num_heads
            return {"attn_output": attn_output, "attn_weights": attn_weights}
        else:
            return attn_output


class RelativeTransformerEncoderLayer(nn.Module):
    """
    d_model: the number of expected features in the input (required).
    nhead: the number of heads in the MultiHeadAttention models (required).
    clipping_threshold: the clipping threshold for relative position embeddings
    dim_feedforward: the dimension of the feedforward network model (default=2048).
    dropout: the dropout value (default=0.1).
    activation: the activation function of the intermediate layer, can be a string
        ("relu" or "gelu") or a unary callable. Default: relu
    layer_norm_eps: the eps value in layer normalization components (default=1e-5).
    norm_first: if ``True``, layer norm is done prior to attention and feedforward
        operations, respectively. Otherwise, it's done after. Default: ``False`` (after).
    """

    # this is some kind of torch jit compiling helper... will also ensure these values don't change
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self,
                 d_model,
                 nhead,
                 pos_encoding="relative",
                 clipping_threshold=3,
                 contact_threshold=7,
                 pdb_fns=None,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=F.relu,
                 layer_norm_eps=1e-5,
                 norm_first=False) -> None:

        self.batch_first = True

        super(RelativeTransformerEncoderLayer, self).__init__()

        self.self_attn = RelativeMultiHeadAttention(d_model, nhead, dropout,
                                                    pos_encoding, clipping_threshold, contact_threshold, pdb_fns)

        # feed forward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, src: Tensor, pdb_fn=None) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), pdb_fn=pdb_fn)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, pdb_fn=None) -> Tensor:
        x = self.self_attn(x, x, x, pdb_fn=pdb_fn)
        if isinstance(x, dict):
            # handle the case where we are returning attention weights
            x = x["attn_output"]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class RelativeTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, reset_params=True):
        super(RelativeTransformerEncoder, self).__init__()
        # using get_clones means all layers have the same initialization
        # this is also a problem in PyTorch's TransformerEncoder implementation, which this is based on
        # todo: PyTorch is changing its transformer API... check up on and see if there is a better way
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        # important because get_clones means all layers have same initialization
        # should recursively reset parameters for all submodules
        if reset_params:
            self.apply(reset_parameters_helper)

    def forward(self, src: Tensor, pdb_fn=None) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(output, pdb_fn=pdb_fn)

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_clones(module, num_clones):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_clones)])


def _inv_dict(d):
    """ helper function for contact map-based position embeddings """
    inv = dict()
    for k, v in d.items():
        # collect dict keys into lists based on value
        inv.setdefault(v, list()).append(k)
    for k, v in inv.items():
        # put in sorted order
        inv[k] = sorted(v)
    return inv


def _combine_d(d, threshold, combined_key):
    """ helper function for contact map-based position embeddings
        d is a dictionary with ints as keys and lists as values.
        for all keys >= threshold, this function combines the values of those keys into a single list """
    out_d = {}
    for k, v in d.items():
        if k < threshold:
            out_d[k] = v
        elif k >= threshold:
            if combined_key not in out_d:
                out_d[combined_key] = v
            else:
                out_d[combined_key] += v
    if combined_key in out_d:
        out_d[combined_key] = sorted(out_d[combined_key])
    return out_d

### structure

# import os
# from os.path import isfile
# from enum import Enum, auto

# import numpy as np
# from scipy.spatial.distance import cdist
# import networkx as nx
# from biopandas.pdb import PandasPdb


class GraphType(Enum):
    LINEAR = auto()
    COMPLETE = auto()
    DISCONNECTED = auto()
    DIST_THRESH = auto()
    DIST_THRESH_SHUFFLED = auto()


def save_graph(g, fn):
    """ Saves graph to file """
    nx.write_gexf(g, fn)


def load_graph(fn):
    """ Loads graph from file """
    g = nx.read_gexf(fn, node_type=int)
    return g


def shuffle_nodes(g, seed=7):
    """ Shuffles the nodes of the given graph and returns a copy of the shuffled graph """
    # get the list of nodes in this graph
    nodes = g.nodes()

    # create a permuted list of nodes
    np.random.seed(seed)
    nodes_shuffled = np.random.permutation(nodes)

    # create a dictionary mapping from old node label to new node label
    mapping = {n: ns for n, ns in zip(nodes, nodes_shuffled)}

    g_shuffled = nx.relabel_nodes(g, mapping, copy=True)

    return g_shuffled


def linear_graph(num_residues):
    """ Creates a linear graph where each node is connected to its sequence neighbor in order """
    g = nx.Graph()
    g.add_nodes_from(np.arange(0, num_residues))
    for i in range(num_residues-1):
        g.add_edge(i, i+1)
    return g


def complete_graph(num_residues):
    """ Creates a graph where each node is connected to all other nodes"""
    g = nx.complete_graph(num_residues)
    return g


def disconnected_graph(num_residues):
    g = nx.Graph()
    g.add_nodes_from(np.arange(0, num_residues))
    return g


def dist_thresh_graph(dist_mtx, threshold):
    """ Creates undirected graph based on a distance threshold """
    g = nx.Graph()
    g.add_nodes_from(np.arange(0, dist_mtx.shape[0]))

    # loop through each residue
    for rn1 in range(len(dist_mtx)):
        # find all residues that are within threshold distance of current
        rns_within_threshold = np.where(dist_mtx[rn1] < threshold)[0]

        # add edges from current residue to those that are within threshold
        for rn2 in rns_within_threshold:
            # don't add self edges
            if rn1 != rn2:
                g.add_edge(rn1, rn2)
    return g


def ordered_adjacency_matrix(g):
    """ returns the adjacency matrix ordered by node label in increasing order as a numpy array """
    node_order = sorted(g.nodes())
    adj_mtx = nx.to_numpy_matrix(g, nodelist=node_order)
    return np.asarray(adj_mtx).astype(np.float32)


def cbeta_distance_matrix(pdb_fn, start=0, end=None):
    # note that start and end are not going by residue number
    # they are going by whatever the listing in the pdb file is

    # read the pdb file into a biopandas object
    ppdb = PandasPdb().read_pdb(pdb_fn)

    # group by residue number
    # important to specify sort=True so that group keys (residue number) are in order
    # the reason is we loop through group keys below, and assume that residues are in order
    # the pandas function has sort=True by default, but we specify it anyway because it is important
    grouped = ppdb.df["ATOM"].groupby("residue_number", sort=True)

    # a list of coords for the cbeta or calpha of each residue
    coords = []

    # loop through each residue and find the coordinates of cbeta
    for i, (residue_number, values) in enumerate(grouped):

        # skip residues not in the range
        end_index = (len(grouped) if end is None else end)
        if i not in range(start, end_index):
            continue

        residue_group = grouped.get_group(residue_number)

        atom_names = residue_group["atom_name"]
        if "CB" in atom_names.values:
            # print("Using CB...")
            atom_name = "CB"
        elif "CA" in atom_names.values:
            # print("Using CA...")
            atom_name = "CA"
        else:
            raise ValueError("Couldn't find CB or CA for residue {}".format(residue_number))

        # get the coordinates of cbeta (or calpha)
        coords.append(
            residue_group[residue_group["atom_name"] == atom_name][["x_coord", "y_coord", "z_coord"]].values[0])

    # stack the coords into a numpy array where each row has the x,y,z coords for a different residue
    coords = np.stack(coords)

    # compute pairwise euclidean distance between all cbetas
    dist_mtx = cdist(coords, coords, metric="euclidean")

    return dist_mtx

def get_neighbors(g, nodes):
    """ returns a list (set) of neighbors of all given nodes """
    neighbors = set()
    for n in nodes:
        neighbors.update(g.neighbors(n))
    return sorted(list(neighbors))


def gen_graph(graph_type, res_dist_mtx, dist_thresh=7, shuffle_seed=7, graph_save_dir=None, save=False):
    """ generate the specified structure graph using the specified residue distance matrix """
    if graph_type is GraphType.LINEAR:
        g = linear_graph(len(res_dist_mtx))
        save_fn = None if not save else os.path.join(graph_save_dir, "linear.graph")

    elif graph_type is GraphType.COMPLETE:
        g = complete_graph(len(res_dist_mtx))
        save_fn = None if not save else os.path.join(graph_save_dir, "complete.graph")

    elif graph_type is GraphType.DISCONNECTED:
        g = disconnected_graph(len(res_dist_mtx))
        save_fn = None if not save else os.path.join(graph_save_dir, "disconnected.graph")

    elif graph_type is GraphType.DIST_THRESH:
        g = dist_thresh_graph(res_dist_mtx, dist_thresh)
        save_fn = None if not save else os.path.join(graph_save_dir, "dist_thresh_{}.graph".format(dist_thresh))

    elif graph_type is GraphType.DIST_THRESH_SHUFFLED:
        g = dist_thresh_graph(res_dist_mtx, dist_thresh)
        g = shuffle_nodes(g, seed=shuffle_seed)
        save_fn = None if not save else \
            os.path.join(graph_save_dir, "dist_thresh_{}_shuffled_r{}.graph".format(dist_thresh, shuffle_seed))

    else:
        raise ValueError("Graph type {} is not implemented".format(graph_type))

    if save:
        if isfile(save_fn):
            print("err: graph already exists: {}. to overwrite, delete the existing file first".format(save_fn))
        else:
            os.makedirs(graph_save_dir, exist_ok=True)
            save_graph(g, save_fn)

    return g

# Huggingface code

class METLConfig(PretrainedConfig):
    IDENT_UUID_MAP = IDENT_UUID_MAP
    UUID_URL_MAP = UUID_URL_MAP
    model_type = "METL"

    def __init__(
            self,
            id:str = None,
            **kwargs,
    ):
        self.id = id
        super().__init__(**kwargs)

class METLModel(PreTrainedModel):
    config_class = METLConfig
    def __init__(self, config:METLConfig):
        super().__init__(config)
        self.model = None
        self.encoder = None
        self.config = config
        
    def forward(self, X, pdb_fn=None):
        if pdb_fn:
            return self.model(X, pdb_fn=pdb_fn)
        return self.model(X)
    
    def load_from_uuid(self, id):
        if id:
            assert id in self.config.UUID_URL_MAP, "ID given does not reference a valid METL model in the IDENT_UUID_MAP"
            self.config.id = id

        self.model, self.encoder = get_from_uuid(self.config.id)

    def load_from_ident(self, id):
        if id:
            id = id.lower()
            assert id in self.config.IDENT_UUID_MAP, "ID given does not reference a valid METL model in the IDENT_UUID_MAP"
            self.config.id = id

        self.model, self.encoder = get_from_ident(self.config.id)

    def get_from_checkpoint(self, checkpoint_path):
        self.model, self.encoder = get_from_checkpoint(checkpoint_path)