# metl-pretrained
Pretrained METL models with minimal dependencies


# Getting started
1. Clone this repository
2. Install the package with `pip install metl-pretrained`
3. Use `model, data_encoder = metl.kThmNaxC()` to load the pre-trained model

# Full example

```python
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
```