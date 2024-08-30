import torch
from typing import Literal
from biopandas.pdb import PandasPdb
import metl.relative_attention as ra
from Bio.SeqUtils import seq1
import os

class ModelEncoder(torch.nn.Module):
    def __init__(self, model, encoder, strict=True, indexing:Literal[0,1] = 0) -> None:
        super(ModelEncoder, self).__init__()

        if indexing != 0 and indexing != 1:
            raise Exception("Indexing must be equal to 0 or to 1.")

        self.model = model
        self.encoder = encoder

        self.indexing = indexing
        self.strict = strict

        self.needs_pdb = self.check_if_pdb_needed(model)

    def check_if_pdb_needed(self, model):
        sequential = next(model.children())
        
        for layer in sequential:
            if isinstance(layer, ra.RelativeTransformerEncoder):
                return True
        return False

    def validate_pdb(self, pdb_file, wt):
        try:
            ppdb = PandasPdb().read_pdb(pdb_file)
        except Exception as e:
            raise Exception(f"{str(e)} \n\n PDB file could not be read by PandasPDB. It may be incorrectly formatted.")

        groups = ppdb.df['ATOM'].groupby('residue_number')
        wt_seq = []
        for group_name, group_data in groups:
            wt_seq.append(seq1(group_data.iloc[0]['residue_name']))
        wildtype = ''.join(wt_seq)

        if self.strict:
            err_str = "Strict mode is on because a METL model that we trained was used. Wildtype and PDB sequeunces must match."
            err_str += " If this is expected behavior, pass strict=False to the load function you used."
            assert wildtype == wt, err_str

    def validate_variants(self, variants, wt):
        wt_len = len(wt)
        for index, variant in enumerate(variants):
            split = variant.split(',')
            for mutation in split:
                from_amino_acid = mutation[0]
                to_amino_acid = mutation[-1]
                location = int(mutation[1:-1])

                errors = []

                if location <= 0 or location >= wt_len-1:
                    error_str = f"The position for the mutation is {location} but it needs to be between 0 "
                    error_str += f"and {len(wt)-1} if 0-based and 1 and {len(wt)} if 1-based."
                    errors.append(error_str)
        
                if wt[location] != from_amino_acid:
                    errors.append(f"Wildtype at position {location} is {wt[location]} but variant had {from_amino_acid}. Check the variant input.")

                if len(errors) != 0:
                    if self.indexing == 1:
                        mutation = f"{from_amino_acid}{location+1}{to_amino_acid}"
                    one_based_variants = self.change_indexing_to(1, variants)

                    raise Exception(f"Invalid mutation {mutation} that is inside variant {one_based_variants[index]}. Errors: {', '.join(errors)}")

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

    def forward(self, wt:str, variants:list[str], pdb_fn:str=None):
        if self.needs_pdb and pdb_fn is None:
            raise Exception("PDB path is required but it was not given. Do you have a PDB file?")

        if pdb_fn:
            pdb_fn = os.path.abspath(pdb_fn)
            self.validate_pdb(pdb_fn, wt)

        if self.indexing == 1:
            variants = self.change_indexing_to(0, variants)

        self.validate_variants(variants, wt)

        encoded_variants = self.encoder.encode_variants(wt, variants)
        
        if pdb_fn:
            pred = self.model(torch.Tensor(encoded_variants), pdb_fn=pdb_fn)
        else:
            pred = self.model(torch.Tensor(encoded_variants))

        return pred