import torch
import metl


def main():
    # this is a 3D RPE model, which requires a PDB file matching the WT sequence
    model, data_encoder = metl.get_from_uuid(uuid="PEkeRuxb")

    # the GFP wild-type sequence
    wt = "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQ" \
         "HDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKN" \
         "GIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

    # some example GFP variants to compute the scores for
    variants = ["E3K,G102S",
                "T36P,S203T,K207R",
                "V10A,D19G,F25S,E113V"]

    encoded_variants = data_encoder.encode_variants(wt, variants)

    # set model to eval mode
    model.eval()
    # no need to compute gradients for inference
    with torch.no_grad():
        predictions = model(torch.tensor(encoded_variants), pdb_fn="./1gfl_cm.pdb")

    print(predictions)


if __name__ == "__main__":
    main()
