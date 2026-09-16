import torch
from typing import Literal
from biopandas.pdb import PandasPdb
from metl.encode import DataEncoder
import metl.relative_attention as ra
from Bio.SeqUtils import seq1
import os
import warnings

class ModelEncoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Sequential, encoder: DataEncoder, 
                 strict:bool=True, indexing:Literal[0,1] = 0) -> None:
        """Wrapper to provide input sanitization and validation for METL model and encoders

        Args:
            model (torch.nn.Sequential): METL model loaded from Zenodo
            encoder (DataEncoder): METL encoder for the above model loaded from Zenodo
            strict (bool, optional): Strict-mode requires PDB files to match WT sequence. Defaults to True.
            indexing (Literal[0,1], optional): What indexing the input variants use. 
                                               Defaults to 0 as METL is programmed to be used with 0 based indexing.

        Raises:
            Exception: Throws an exception if the indexing is invalid
        """

        super(ModelEncoder, self).__init__()

        if indexing != 0 and indexing != 1:
            raise AssertionError("Indexing must be equal to 0 or to 1.")

        self.model = model
        self.encoder = encoder

        self.indexing = indexing
        self.strict = strict

        self.needs_pdb = self.check_if_pdb_needed(model)

    def check_if_pdb_needed(self, model: torch.nn.Sequential):

        sequential = next(model.children())
        
        for layer in sequential:
            if isinstance(layer, ra.RelativeTransformerEncoder):
                return True
        return False

    def validate_pdb(self, pdb_file:str , wt: str|list[str]):
        """
        When validating a PDB, it is possible that the PDB file and wild type (wt) passed will differ. 
        Strict raises an exception if this occurs, otherwise this potential error is not checked.
        Strict is off by default when loading from a checkpoint file, and on when loading models from Zenodo.

        Args:
            pdb_file (str): The path to the PDB file
            wt (str): The string representing the wild type sequence

        Raises:
            Exception: Raises exceptions for multi-chain input PDB files or files unable to be loaded by pandaspdb 
        """

        # Check valid
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")  # All warnings triggered

                # Code that might trigger warnings
                PandasPdb().read_pdb(pdb_file)

                if len(w) > 0:
                    warns = [str(warn.message) for warn in w]
                    joined_warnings = "\n".join(warns)
                    raise AssertionError(f"Pandas PDB is giving a warning, this usually means a PDB file is wrong: \n{joined_warnings}")
                    
            ppdb = PandasPdb().read_pdb(pdb_file)
        except Exception as e:
            raise ValueError(f"{e} \n\n PDB file could not be read by PandasPDB. It may be incorrectly formatted.")

        # Check multi-chain 
        num_chains = ppdb.df['ATOM'].groupby('chain_id').size().size
        
        if num_chains != 1:
            raise ValueError(f"PDB file has {num_chains} chains. METL only supports single chain PDB files.")

        groups = ppdb.df['ATOM'].groupby('residue_number')
        wt_seq = []
        for group_name, group_data in groups:
            wt_seq.append(seq1(group_data.iloc[0]['residue_name']))
        wildtype = ''.join(wt_seq)

        if self.strict and isinstance(wt, str):
            err_str = "Strict mode is on because a METL model that we trained was used. Wildtype and PDB sequences must match."
            err_str += " To ignore the sequence mismatch, pass strict=False to the load function you used."
            assert wildtype == wt, err_str
        elif isinstance(wt, list):
            assert all([isinstance(AASeq, str) for AASeq in wt]), "One or more sequence in the list of sequences you passed was not a string."
            # WT seqs must be the same length
            for seq in wt:
                err_str = "One of the sequences in the list of sequences you passed was not the same length as the "
                err_str += "first sequence. All sequences must be the same length."
                assert len(seq) == len(wt[0]), err_str

    def validate_variants(self, variants, wt):
        """
        Variants much be validated only after conversion to 0 based!
        """
        wt_len = len(wt)
        for index, variant in enumerate(variants):
            split = variant.split(',')
            for mutation in split:
                from_amino_acid = mutation[0]
                to_amino_acid = mutation[-1]
                location = int(mutation[1:-1])

                error = None

                if location < 0 or location >= wt_len:
                    error_str = f"The position for the mutation is {location} but it needs to be between 0 "
                    error_str += f"and {len(wt)-1} if 0-based and 1 and {len(wt)} if 1-based."
                    error = error_str
                elif wt[location] != from_amino_acid:
                    error = f"Wildtype at position {location} is {wt[location]} but variant had {from_amino_acid}. Check the variant input."

                if error is not None:
                    if self.indexing == 1:
                        mutation = f"{from_amino_acid}{location+1}{to_amino_acid}"
                    one_based_variants = self.change_indexing_to(1, variants)

                    raise AssertionError(f"Invalid mutation {mutation} that is inside variant {one_based_variants[index]}. Error: {error}")

    def change_indexing_to(self, indexing, variants):
        changed_based_variants = []
        for variant in variants:
            split = variant.split(',')
            variant_strings = []
            for mutation in split:
                from_amino_acid = mutation[0]
                to_amino_acid = mutation[-1]
                location = int(mutation[1:-1])
                
                if indexing == 0:
                    location = location-1
                else:
                    location = location + 1

                variant_strings.append(f'{from_amino_acid}{location}{to_amino_acid}')
            changed_based_variants.append(",".join(variant_strings))
        
        return changed_based_variants

    def forward(self, wt:str|list[str], variants:list[str]=None, pdb_fn:str=None):
        if isinstance(wt, str) and self.needs_pdb and pdb_fn is None:
            raise AssertionError("PDB path is required but it was not given. Do you have a PDB file?")

        if pdb_fn:
            pdb_fn = os.path.abspath(os.path.expanduser(pdb_fn))
            self.validate_pdb(pdb_fn, wt)
        
        encoded_variants = None
        if isinstance(wt, list):
            # Reusing this variable so we can have a simpler program flow
            assert all([isinstance(AASeq, str) for AASeq in wt]), "All sequences in wt must be type str"
            encoded_variants = self.encoder.encode_sequences(wt)

        if variants is not None:
            if isinstance(variants, str):
                assert ValueError("Variants is just a string. Did you forget to use pdb_fn by keyword argument, when trying to pass in multiple sequences?")
            if self.indexing == 1:
                variants = self.change_indexing_to(0, variants)

            self.validate_variants(variants, wt)

            encoded_variants = self.encoder.encode_variants(wt, variants)
        
        assert encoded_variants is not None, "Encoding did not happen. If wt is a string, you must pass variants. If wt is a list, you can not pass variants."

        if pdb_fn:
            pred = self.model(torch.tensor(encoded_variants), pdb_fn=pdb_fn)
        else:
            pred = self.model(torch.tensor(encoded_variants))

        return pred