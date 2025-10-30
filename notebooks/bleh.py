from transformers import AutoModel
import torch

metl = AutoModel.from_pretrained('gitter-lab/METL', trust_remote_code=True)

model = "metl-l-2m-3d-gb1"
wt = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
variants = '["T17P,T54F", "V28L,F51A"]'
pdb_path = './2qmt_p.pdb'

metl.load_from_ident(model)
metl.eval()

with torch.no_grad():   
    predictions = metl(wt, variants, pdb_path)