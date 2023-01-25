import torch
import metl


def main():
    ckpt_fn = "/Users/sg/PycharmProjects/RosettaTL/output/htcondor_runs/target/dev/condor_2023-01-25_12-58-18_avgfp_metl_local_for_bryce/run_output/training_logs/bcEoygY3/checkpoints/bcEoygY3.pt"
    model, data_encoder = metl.get_from_checkpoint(ckpt_fn)

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
