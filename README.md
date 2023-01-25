# Pretrained METL models
This repository contains pretrained METL models with minimal dependencies.

# Getting started
1. Create a conda environment (or use existing one): `conda create --name myenv python=3.9`
2. Activate conda environment `conda activate myenv`
3. Clone this repository
4. Navigate to the cloned repository `cd metl-pretrained`
5. Install the package with `pip install .`
6. Import the package in your script with `import metl`
7. Load a pretrained model using `model, data_encoder = metl.get_from_uuid(uuid)` or one of the other loading functions (see examples below)
    - `model` is a PyTorch model loaded with the pre-trained weights
    - `data_encoder` is a helper object that can be used to encode sequences and variants to be fed into the model

# Available models
Model checkpoints are available to download from Box.
Once you have a checkpoint downloaded, you can load it into a PyTorch model using `metl.get_from_checkpoint()`.
Alternatively, you can use `metl.get_from_uuid()` or `metl.get_from_ident()` to automatically download, cache, and load the model based on the model identifier or UUID.
See the examples below.

## Source models
Source models predict Rosetta energy terms.

| Identifier         | UUID       | Description           | Download                                                                 |
|--------------------|------------|-----------------------|--------------------------------------------------------------------------|
| `METL-G-20M-1D`    | `D72M9aEp` | METL-G (20M, 1D)      | [Download](https://uwmadison.box.com/s/dj1b605pqmkep4eard45p75xvlk5nvpl) |
| `METL-G-20M-3D`    | `Nr9zCKpR` | METL-G (20M, 3D)      | [Download](https://uwmadison.box.com/s/x03hzg0rvtomj3n47fkroahn7k38wu82) |
| `METL-L-2M-1D-GFP` | `8gMPQJy4` | METL-L (2M, 1D) [GFP] | [Download](https://uwmadison.box.com/s/2fyd0ecft0dlvfo29hvfina0fwcq0y46) |
| `METL-L-2M-3D-GFP` | `Hr4GNHws` | METL-L (2M, 1D) [GFP] | [Download](https://uwmadison.box.com/s/fveywo9t1jtbsl3qrhjcthgd3ltwfrnp) |
 
## Target models
Target models are fine-tuned source models that predict functional scores from experimental sequence-function data.

| Identifier | UUID       | Description                                                             | Download                                                                 |
|------------|------------|-------------------------------------------------------------------------| ------------------------------------------------------------------------ |
| None       | `bcEoygY3` | `METL-L-2M-1D-GFP` fine-tuned on 80 examples from the avGFP DMS dataset | [Download](https://uwmadison.box.com/s/lgjj1sxctx1rkbuvp8nvzxuq5g5l2z1g) |
| None       | `54etfaYj` | `METL-L-2M-3D-GFP` fine-tuned on 80 examples from the avGFP DMS dataset | [Download](https://uwmadison.box.com/s/rrefcranfmqrc9ghj6mu51abmkdb2mth) |


# Examples

## METL source model

METL source models are assigned identifiers that can be used to load the model with `metl.get_from_ident()`. 

This example:
- Automatically downloads and caches `METL-G-20M-1D` using `metl.get_from_ident("metl-g-20m-1d")`.
- Encodes a pair of dummy amino acid sequences using `data_encoder.encode_sequences()`.
- Runs the sequences through the model and prints the predicted Rosetta energies.

_Todo: show how to extract the METL representation at different layers of the network_ 

```python
import metl
import torch

model, data_encoder = metl.get_from_ident("metl-g-20m-1d")

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

# METL target model

METL target models can be loaded using the model's UUID and `metl.get_from_uuid()`.

This example:
- Automatically downloads and caches `bcEoygY3` using `metl.get_from_uuid(uuid="bcEoygY3")`.
- Encodes several variants in specified in variant notation. A wild-type sequence is needed to encode variants.
- Runs the sequences through the model and prints the predicted DMS scores.

```python
import metl
import torch

model, data_encoder = metl.get_from_uuid(uuid="bcEoygY3")

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

```
