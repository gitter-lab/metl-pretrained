import metl
model, encoder = metl.get_from_ident("metl-l-2m-3d-gb1")

wt = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
variants = ["T17P,T54F", "V28L,F51A"]

encoded_variants = encoder.encode_variants(wt, variants)

print(model(encoded_variants, pdb_fn = './pdbs/2qmt_p.pdb'))