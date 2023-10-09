import metl
import torch


def main():
    model, data_encoder = metl.get_from_ident("METL-L-2M-3D-GB1")

    # the GB1 WT sequence
    wt = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

    # some example GB1 variants for which to compute the METL-Local Rosetta scores estimates
    variants = ["T17P,T54F",
                "V28L,F51A",
                "T17P,V28L,F51A,T54F"]

    encoded_variants = data_encoder.encode_variants(wt, variants)

    # set model to eval mode
    model.eval()
    # no need to compute gradients for inference
    with torch.no_grad():
        predictions = model(torch.tensor(encoded_variants), pdb_fn="../pdbs/2qmt_p.pdb")
    print(predictions)

    # can also input full sequences
    sequences = ["MPYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
                 "MPAKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
                 "MGEKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"]
    encoded_sequences = data_encoder.encode_sequences(sequences)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(encoded_sequences), pdb_fn="../pdbs/2qmt_p.pdb")
    print(predictions)

    # can also use the 1D model which doesn't require a PDB file
    model, data_encoder = metl.get_from_ident("METL-L-2M-1D-GB1")
    variants = ["T17P,T54F",
                "V28L,F51A",
                "T17P,V28L,F51A,T54F"]
    encoded_variants = data_encoder.encode_variants(wt, variants)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(encoded_variants))
    print(predictions)


if __name__ == "__main__":
    main()
