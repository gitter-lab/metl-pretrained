import metl
import torch


def main():
    model, data_encoder = metl.get_from_ident("METL-L-2M-3D-GB1")

    # the GB1 WT sequence
    wt = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

    # some example GB1 variants to compute the Rosetta scores for
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


if __name__ == "__main__":
    main()
