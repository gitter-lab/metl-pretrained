from transformers import AutoModel

def main():
    metl = AutoModel.from_pretrained('gitter-lab/METL', trust_remote_code=True)
    start = "# @param ["
    metl_keys = [f'"{key}"' for key in metl.config.IDENT_UUID_MAP.keys()]
    keys = ','.join(metl_keys)
    end = f'{keys}]'
    print(start + end)

if __name__ == "__main__":
    main()