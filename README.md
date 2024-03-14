# Pretrained METL models
[![GitHub Actions](https://github.com/gitter-lab/metl-pretrained/actions/workflows/test.yml/badge.svg)](https://github.com/gitter-lab/metl-pretrained/actions/workflows/test.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10819499.svg)](https://zenodo.org/doi/10.5281/zenodo.10819499)

This repository contains pretrained METL models with minimal dependencies.
For more information, please see the [metl](https://github.com/gitter-lab/metl) repository and our manuscript:

**Biophysics-based protein language models for protein engineering**.  
Sam Gelman, Bryce Johnson, Chase Freschlin, Sameer D'Costa, Anthony Gitter<sup>+</sup>, Philip A Romero<sup>+</sup>.  
*bioRxiv*, 2024.  
<sup>+</sup> denotes equal contribution.

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
Box is a temporary storage location, and the models will be moved to stable, archival storage in the future.
Once you have a checkpoint downloaded, you can load it into a PyTorch model using `metl.get_from_checkpoint()`.
Alternatively, you can use `metl.get_from_uuid()` or `metl.get_from_ident()` to automatically download, cache, and load the model based on the model identifier or UUID.
See the examples below.

## Source models
Source models predict Rosetta energy terms.

### Global source models
| Identifier      | UUID       | Params | RPE | Output           | Description | Download                                                                                |
|-----------------|------------|--------|-----|------------------|-------------|-----------------------------------------------------------------------------------------|
| `METL-G-20M-1D` | `D72M9aEp` | 20M    | 1D  | Rosetta energies | METL-G      | [Download](https://uwmadison.box.com/s/dj1b605pqmkep4eard45p75xvlk5nvpl)                |
| `METL-G-20M-3D` | `Nr9zCKpR` | 20M    | 3D  | Rosetta energies | METL-G      | [Download](https://uwmadison.box.com/s/x03hzg0rvtomj3n47fkroahn7k38wu82)                |
| `METL-G-50M-1D` | `auKdzzwX` | 50M    | 1D  | Rosetta energies | METL-G      | [Download](https://uwmadison.box.com/shared/static/ir4xmq1g44w9a7o1xdv94teonicyzoht.pt) |
| `METL-G-50M-3D` | `6PSAzdfv` | 50M    | 3D  | Rosetta energies | METL-G      | [Download](https://uwmadison.box.com/shared/static/fp1tqbuad95bfe00djpb5lcb8e4as8f1.pt) |


### Local source models
| Identifier         | UUID       | Protein | Params | RPE | Output           | Description | Download                                                                                |
|--------------------|------------|---------|--------|-----|------------------|-------------|-----------------------------------------------------------------------------------------|
| `METL-L-2M-1D-GFP` | `8gMPQJy4` | avGFP   | 2M     | 1D  | Rosetta energies | METL-L      | [Download](https://uwmadison.box.com/s/2fyd0ecft0dlvfo29hvfina0fwcq0y46)                |
| `METL-L-2M-3D-GFP` | `Hr4GNHws` | avGFP   | 2M     | 3D  | Rosetta energies | METL-L      | [Download](https://uwmadison.box.com/s/fveywo9t1jtbsl3qrhjcthgd3ltwfrnp)                |
 | `METL-L-2M-1D-GB1` | `DMfkjVzT` | GB1     | 2M     | 1D  | Rosetta energies | METL-L      | [Download](https://uwmadison.box.com/shared/static/u3p7hi9vb9p4civxzk9puc0sb48aisei.pt) |
| `METL-L-2M-3D-GB1` | `epegcFiH` | GB1     | 2M     | 3D  | Rosetta energies | METL-L      | [Download](https://uwmadison.box.com/shared/static/3bcoqgmij5tsfcuggpum4i3ovgqjbzju.pt) |

These models will output a length 55 vector corresponding to the following energy terms (in order):
<details>
  <summary>
    Expand to see energy terms
  </summary>

```
total_score
fa_atr
fa_dun
fa_elec
fa_intra_rep
fa_intra_sol_xover4
fa_rep
fa_sol
hbond_bb_sc
hbond_lr_bb
hbond_sc
hbond_sr_bb
lk_ball_wtd
omega
p_aa_pp
pro_close
rama_prepro
ref
yhh_planarity
buried_all
buried_np
contact_all
contact_buried_core
contact_buried_core_boundary
degree
degree_core
degree_core_boundary
exposed_hydrophobics
exposed_np_AFIMLWVY
exposed_polars
exposed_total
one_core_each
pack
res_count_buried_core
res_count_buried_core_boundary
res_count_buried_np_core
res_count_buried_np_core_boundary
ss_contributes_core
ss_mis
total_hydrophobic
total_hydrophobic_AFILMVWY
total_sasa
two_core_each
unsat_hbond
centroid_total_score
cbeta
cenpack
env
hs_pair
pair
rg
rsigma
sheet
ss_pair
vdw
```
</details>


### Function-specific source models for GB1

The GB1 experimental data measured the binding interaction between GB1 variants and Immunoglobulin G (IgG). 
To match this experimentally characterized function, we implemented a Rosetta pipeline to model the GB1-IgG complex and compute 17 attributes related to energy changes upon binding.
We pretrained a standard METL-Local model and a modified METL-Bind model, which additionally incorporates the IgG binding attributes into its pretraining tasks.

| Identifier                     | UUID       | Protein | Params | RPE | Output                              | Description                                                                                                                                                                       | Download                                                                                |
|--------------------------------|------------|---------|--------|-----|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `METL-BIND-2M-3D-GB1-STANDARD` | `K6mw24Rg` | GB1     | 2M     | 3D  | Standard Rosetta energies           | Trained for the function-specific synthetic data experiment, but only trained on the standard energy terms, to use as a baseline. Should perform similarly to `METL-L-2M-3D-GB1`. | [Download](https://uwmadison.box.com/shared/static/9rqrlvivwrcs8o8s1koncywslgypcssw.pt)                |
| `METL-BIND-2M-3D-GB1-BINDING`  | `Bo5wn2SG` | GB1     | 2M     | 3D  | Standard + binding Rosetta energies | Trained on both the standard energy terms and the binding-specific energy terms.                                                                                                  | [Download](https://uwmadison.box.com/shared/static/y8vvq3yz24qbu6crws3um1yoi0hv79tp.pt)                |


`METL-BIND-2M-3D-GB1-BINDING` predicts the standard energy terms listed above as well as the following binding energy terms (in order):

<details>
  <summary>
    Expand to see binding energy terms
  </summary>

```
complex_normalized
dG_cross
dG_cross/dSASAx100
dG_separated
dG_separated/dSASAx100
dSASA_hphobic
dSASA_int
dSASA_polar
delta_unsatHbonds
hbond_E_fraction
hbonds_int
nres_int
per_residue_energy_int
side1_normalized
side1_score
side2_normalized
side2_score
```
</details>

## Target models
Target models are fine-tuned source models that predict functional scores from experimental sequence-function data.

| DMS Dataset | Identifier | UUID       | RPE | Output           | Description                                                                                                                                                        | Download                                                                                |
|:------------|------------|------------|-----|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| avGFP       | `None`     | `YoQkzoLD` | 1D  | Functional score | The `METL-L-2M-1D-GFP` model, fine-tuned on 64 examples from the avGFP DMS dataset. This model was used for the GFP design experiment described in the manuscript. | [Download](https://uwmadison.box.com/shared/static/6rcwwl7mcbt4tgmemhxodavkx66ihi9h.pt) |
| avGFP       | `None`     | `PEkeRuxb` | 3D  | Functional score | The `METL-L-2M-3D-GFP` model, fine-tuned on 64 examples from the avGFP DMS dataset. This model was used for the GFP design experiment described in the manuscript. | [Download](https://uwmadison.box.com/shared/static/spzvqyct4d6qyfjxqqsi4ygxnq649p01.pt) |


# 3D Relative Position Embeddings

METL uses relative position embeddings (RPEs) based on 3D protein structure. 
The implementation of relative position embeddings is similar to the original paper by [Shaw et al](https://aclanthology.org/N18-2074/).
However, instead of using the default 1D sequence-based distances, we calculate relative distances based on a graph of the 3D protein structure.
These 3D RPEs enable the transformer to use 3D distances between amino acid residues as the positional signal when calculating attention.
When using 3D RPEs, the model requires a protein structure in the form of a PDB file, corresponding to the wild-type protein or base protein of the input variant sequence.

Our testing showed that 3D RPEs improve performance for METL-Global models but do not make a difference for METL-Local models.
We provide both 1D and 3D models in this repository. The 1D models do not require the PDB structure as an additional input.

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

If you are using a model with 3D relative position embeddings, you will need to provide the PDB structure of the wild-type or base protein.

```
predictions = model(torch.tensor(encoded_seqs), pdb_fn="../path/to/file.pdb")
```


# METL target model

METL target models can be loaded using the model's UUID and `metl.get_from_uuid()`.

This example:
- Automatically downloads and caches `bcEoygY3` using `metl.get_from_uuid(uuid="bcEoygY3")`.
- Encodes several variants specified in variant notation. A wild-type sequence is needed to encode variants.
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
