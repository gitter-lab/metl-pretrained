# metl-pretrained
Pretrained METL models with minimal dependencies


# Getting started
1. Create a conda environment (or use existing one): `conda create --name metl-test python=3`
2. Activate conda environment `conda activate metl-test`
3. Clone this repository
4. Navigate to the cloned repository `cd metl-pretrained`
5. Install the package with `pip install .`
6. Import the package in your script with `import metl` (example below)
7. Use `model, data_encoder = metl.kThmNaxC()` to load the pre-trained model (example below)

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