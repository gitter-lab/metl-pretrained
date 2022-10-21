""" Encodes data in different formats """
from enum import Enum, auto

import numpy as np


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
