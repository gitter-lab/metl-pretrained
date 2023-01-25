import torch
import metl


def main():
    # "bcEoygY3" is a METL-L (2M, 1D) [GFP] model that was fine-tuned on 80 examples from the avGFP DMS dataset
    model, data_encoder = metl.get_uuid(uuid="bcEoygY3")

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
        predictions = model(torch.tensor(encoded_variants))

    print(predictions)


if __name__ == "__main__":
    main()
