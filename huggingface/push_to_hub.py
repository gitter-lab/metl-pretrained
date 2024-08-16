from huggingface.huggingface_wrapper import METLConfig, METLModel
from huggingface_hub import login
import os
from transformers import AutoModel, AutoConfig
import torch

def main():
    API_KEY = os.getenv('HF_TOKEN')
    login(API_KEY)

    config = METLConfig()
    model = METLModel(config)
    model.model = torch.nn.Linear(1, 1)

    AutoConfig.register("METL", METLConfig)
    AutoModel.register(METLConfig, METLModel)

    model.register_for_auto_class()
    config.register_for_auto_class()

    model.push_to_hub('gitter-lab/METL')
    config.push_to_hub('gitter-lab/METL')

if __name__ == "__main__":
    main()