"""
This file contains the actual wrapper for METL. 
Above the delimiter for this file: #\$\> we have included imports and shell functions 
which prevent python (and other linters) from complaining this file has erros. 
"""


from transformers import PretrainedConfig, PreTrainedModel

def get_from_uuid():
    pass

def get_from_ident():
    pass

def get_from_checkpoint():
    pass

IDENT_UUID_MAP = ""
UUID_URL_MAP = ""

# Chop The above off. 

#$>
# Huggingface code

class METLConfig(PretrainedConfig):
    IDENT_UUID_MAP = IDENT_UUID_MAP
    UUID_URL_MAP = UUID_URL_MAP
    model_type = "METL"

    def __init__(
            self,
            id:str = None,
            **kwargs,
    ):
        self.id = id
        super().__init__(**kwargs)

class METLModel(PreTrainedModel):
    config_class = METLConfig
    def __init__(self, config:METLConfig):
        super().__init__(config)
        self.model = None
        self.encoder = None
        self.config = config
        
    def forward(self, X, pdb_fn=None):
        if pdb_fn:
            return self.model(X, pdb_fn=pdb_fn)
        return self.model(X)
    
    def load_from_uuid(self, id):
        if id:
            assert id in self.config.UUID_URL_MAP, "ID given does not reference a valid METL model in the IDENT_UUID_MAP"
            self.config.id = id

        self.model, self.encoder = get_from_uuid(self.config.id)

    def load_from_ident(self, id):
        if id:
            id = id.lower()
            assert id in self.config.IDENT_UUID_MAP, "ID given does not reference a valid METL model in the IDENT_UUID_MAP"
            self.config.id = id

        self.model, self.encoder = get_from_ident(self.config.id)

    def get_from_checkpoint(self, checkpoint_path):
        self.model, self.encoder = get_from_checkpoint(checkpoint_path)
