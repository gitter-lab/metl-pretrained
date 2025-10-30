import metl
import torch
import pytest

##### Useful Global Variables used across tests
MULTI_CHAIN_PDB = '../pdbs/8hgs_multi_chain.pdb'

SINGLE_CHAIN_PDB = '../pdbs/2qmt_p.pdb'
SINGLE_CHAIN_MODEL = 'metl-l-2m-3d-gb1'
SINGLE_CHAIN_VALID_WT = 'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
SINGLE_CHAIN_INVALID_WT = 'VQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE' # Index 0, M to V
SINGLE_CHAIN_VALID_VARIANTS = ["T17P,T54F", "V28L,D46A"]

INVALID_SINGLE_CHAIN_PDB = '../pdbs/invalid_pdb.pdb'

##### PDB FILE TESTS #####

def test_multichain_pdb():
    def call_model():
        model = metl.get_from_ident(SINGLE_CHAIN_MODEL)
        model.eval()
        with torch.no_grad():
            model(SINGLE_CHAIN_VALID_WT, SINGLE_CHAIN_VALID_VARIANTS, MULTI_CHAIN_PDB)
    
    with pytest.raises(ValueError) as excinfo:
        call_model()
    assert "chains" in str(excinfo.value)
    
def test_invalid_pdb():
    def call_model():
        model = metl.get_from_ident(SINGLE_CHAIN_MODEL)
        model.eval()
        with torch.no_grad():
            model(SINGLE_CHAIN_VALID_WT, SINGLE_CHAIN_VALID_VARIANTS, INVALID_SINGLE_CHAIN_PDB)
    
    with pytest.raises(ValueError) as excinfo:
        call_model()
    assert "PDB file could not be read by PandasPDB" in str(excinfo.value)

def test_check_require_pdb():
    def call_model():
        model = metl.get_from_ident(SINGLE_CHAIN_MODEL)
        model.eval()
        with torch.no_grad():
            model(SINGLE_CHAIN_VALID_WT, SINGLE_CHAIN_VALID_VARIANTS)
    
    with pytest.raises(AssertionError) as excinfo:
        call_model()
    assert "PDB path is required" in str(excinfo.value)

##### STICT MODE TESTING #####

def test_invalid_wildtype_strict_on():
    def call_model():
        model = metl.get_from_ident(SINGLE_CHAIN_MODEL)
        model.eval()
        with torch.no_grad():
            model(SINGLE_CHAIN_INVALID_WT, SINGLE_CHAIN_VALID_VARIANTS, SINGLE_CHAIN_PDB)
    
    with pytest.raises(AssertionError) as excinfo:
        call_model()
    assert "Strict mode is on" in str(excinfo.value)

def test_invalid_wildtype_strict_off():
    model = metl.get_from_ident(SINGLE_CHAIN_MODEL, strict=False)
    model.eval()
    with torch.no_grad():
        model(SINGLE_CHAIN_INVALID_WT, SINGLE_CHAIN_VALID_VARIANTS, SINGLE_CHAIN_PDB)
    assert True
    print("Invalid wilidtype allowed with strict mode off")    

##### INDEXING TESTING #####

def test_valid_indexing():
    one_based_variants = ["M1K,E56K"]
    zero_based_variants = ["M0K,E55K"]

    model_1 = metl.get_from_ident(SINGLE_CHAIN_MODEL, indexing=1)
    model_0 = metl.get_from_ident(SINGLE_CHAIN_MODEL, indexing=0)
    model_1.eval()
    model_0.eval()
    with torch.no_grad():
        model_0_embed = model_0(SINGLE_CHAIN_VALID_WT, zero_based_variants, SINGLE_CHAIN_PDB)
        model_1_embed = model_1(SINGLE_CHAIN_VALID_WT, one_based_variants, SINGLE_CHAIN_PDB)
    
    assert torch.all(torch.eq(model_0_embed,model_1_embed)), "0 based vs 1 based embeddings are not the same."

def test_invalid_indexing():
    with pytest.raises(AssertionError):
        metl.get_from_ident(SINGLE_CHAIN_MODEL, indexing=3)
        metl.get_from_ident(SINGLE_CHAIN_MODEL, indexing=-1)

##### MUTATION TESTING #####

def test_out_of_bounds_mutation():
    out_of_bounds_mutation = ["E56K"] # Out of bounds if 1 based
    model = metl.get_from_ident(SINGLE_CHAIN_MODEL)
    with pytest.raises(AssertionError) as excinfo:
        model(SINGLE_CHAIN_VALID_WT, out_of_bounds_mutation, SINGLE_CHAIN_PDB)
    assert "The position for the mutation" in str(excinfo.value)

def test_start_end_mutations():
    start_end_mutations = ["M0K,E55K"]
    model = metl.get_from_ident(SINGLE_CHAIN_MODEL)
    model(SINGLE_CHAIN_VALID_WT, start_end_mutations, SINGLE_CHAIN_PDB)

def test_wt_amino_acid_mismatch():
    """
    Metl expects that the first amino acid (AA) of the mutation AA1 pos AA2 (AA1)
    Is the same as the wild type. This test makes sure the check is working for that. 
    """
    mutation = ["V0K"] 
    model = metl.get_from_ident(SINGLE_CHAIN_MODEL)
    with pytest.raises(AssertionError) as excinfo:
        model(SINGLE_CHAIN_VALID_WT, mutation, SINGLE_CHAIN_PDB)
    print(str(excinfo.value))
    assert "Wildtype at position" in str(excinfo.value)

# We don't check that the mutation is a valid amino acid. I'll leave that for later, I guess.

##### MODEL PREDICTION TESTING #####

# These are just the old tests but asserting they are the same as the old API

def test_global_pred():
    model, data_encoder = metl.get_from_ident("metl-g-20m-1d", raw=True)
    model_api = metl.get_from_ident("metl-g-20m-1d")

    # make sure all the sequences are the same length
    amino_acid_sequences = ["SMART", "MAGIC"]
    encoded_seqs = data_encoder.encode_sequences(amino_acid_sequences)

    # set model to eval mode
    model.eval()
    model_api.eval()
    # no need to compute gradients for inference
    with torch.no_grad():
        predictions = model(torch.tensor(encoded_seqs))
        api_pred = model_api(amino_acid_sequences)
        assert torch.all(torch.eq(predictions, api_pred))

def test_global_variants_pred():
    # "YoQkzoLD" is a METL-L (2M, 1D) [GFP] model that was fine-tuned on 64 examples from the avGFP DMS dataset
    model, data_encoder = metl.get_from_uuid(uuid="YoQkzoLD", raw=True)
    model_api = metl.get_from_uuid(uuid="YoQkzoLD")

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
    model_api.eval()
    # no need to compute gradients for inference
    with torch.no_grad():
        predictions = model(torch.tensor(encoded_variants))
        pred_api = model_api(wt, variants)
        assert torch.all(torch.eq(predictions, pred_api))

def test_local_1D():
    model, data_encoder = metl.get_from_ident("METL-L-2M-3D-GB1", raw=True)
    model_api = metl.get_from_ident("METL-L-2M-3D-GB1")
    # the GB1 WT sequence
    wt = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

    # some example GB1 variants for which to compute the METL-Local Rosetta scores estimates
    variants = ["T17P,T54F",
                "V28L,F51A",
                "T17P,V28L,F51A,T54F"]

    encoded_variants = data_encoder.encode_variants(wt, variants)

    # set model to eval mode
    model.eval()
    model_api.eval()
    # no need to compute gradients for inference
    with torch.no_grad():
        predictions = model(torch.tensor(encoded_variants), pdb_fn="../pdbs/2qmt_p.pdb")
        pred_api = model_api(wt, variants, "../pdbs/2qmt_p.pdb")
        assert torch.all(torch.eq(predictions, pred_api))

    # can also input full sequences
    sequences = ["MPYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
                 "MPAKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
                 "MGEKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"]
    encoded_sequences = data_encoder.encode_sequences(sequences)
    
    model.eval()
    model_api.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(encoded_sequences), pdb_fn="../pdbs/2qmt_p.pdb")
        # Have to use keyword argument if not passing variants but am passing a pdb_fn
        pred_api = model_api(sequences, pdb_fn="../pdbs/2qmt_p.pdb")
        assert torch.all(torch.eq(predictions, pred_api))

    # can also use the 1D model which doesn't require a PDB file
    model, data_encoder = metl.get_from_ident("METL-L-2M-1D-GB1", raw=True)
    model_api = metl.get_from_ident("METL-L-2M-1D-GB1")
    variants = ["T17P,T54F",
                "V28L,F51A",
                "T17P,V28L,F51A,T54F"]
    encoded_variants = data_encoder.encode_variants(wt, variants)
    
    model.eval()
    model_api.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(encoded_variants))
        pred_api = model_api(wt, variants)
        assert torch.all(torch.eq(predictions, pred_api))

def test_local_3D():
    # this is a 3D RPE model, which requires a PDB file matching the WT sequence
    model, data_encoder = metl.get_from_uuid(uuid="PEkeRuxb", raw=True)
    model_api = metl.get_from_uuid(uuid="PEkeRuxb")
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
    model_api.eval()
    # no need to compute gradients for inference
    with torch.no_grad():
        predictions = model(torch.tensor(encoded_variants), pdb_fn="../pdbs/1gfl_cm.pdb")
        pred_api = model_api(wt, variants, "../pdbs/1gfl_cm.pdb")
        assert torch.all(torch.eq(predictions, pred_api))


