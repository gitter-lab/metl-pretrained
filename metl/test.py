import metl
import torch

# kThmNaxC --> sequence-based position embedding (no structure needed) 
# VnsMZTkf --> structure-based position embedding (need PDB)
model, data_encoder = metl.kThmNaxC()

# these are amino acid sequences
# make sure all the sequences are the same length
dummy_sequences = ["SMART", "MAGIC"]
encoded_seqs = data_encoder.encode_sequences(dummy_sequences)

# set model to eval mode
model.eval()
# no need to compute gradients for inference
with torch.no_grad():
    predictions = model(torch.tensor(encoded_seqs))

print(predictions)
