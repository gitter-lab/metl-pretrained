import metl
import torch


def main():
    model, data_encoder = metl.get_from_ident("metl-g-20m-1d")

    # make sure all the sequences are the same length
    amino_acid_sequences = ["SMART", "MAGIC"]
    encoded_seqs = data_encoder.encode_sequences(amino_acid_sequences)

    # set model to eval mode
    model.eval()
    # no need to compute gradients for inference
    with torch.no_grad():
        predictions = model(torch.tensor(encoded_seqs))

    print(predictions)


if __name__ == "__main__":
    main()
