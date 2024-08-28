"""
This script combines all of the files in the metl directory into one file so that it can be uploaded automatically to huggingface.

Files ending with _.py and that contain test in the filename will not be included. This script automatically generates the required imports from the files as well.

Regardless of changes to metl, as long as necessary files that may be added don't contain test or _.py, this should work as intended.
"""

import argparse
import os

def main(output_path: str):
    imports = set()
    code = []
    metl_imports = set()
    for file in os.listdir('./metl'):
        if '.py' in file and '_.py' not in file and 'test' not in file:
            with open(f'./metl/{file}', 'r') as f:
                file_text = f.readlines()
                for line in file_text:
                    line_for_compare = line.strip()
                    if 'import ' in line_for_compare and 'metl.' not in line_for_compare:
                        imports.add(line_for_compare)
                    elif 'import ' in line_for_compare and 'metl.' in line_for_compare:
                        if 'as' in line_for_compare:
                            metl_imports.add(line_for_compare)
                    else:
                        code.append(line[:-1])

    code = '\n'.join(code)
    imports = '\n'.join(imports)

    for line in metl_imports:
        import_name = line.split('as')[-1].strip()
        code = code.replace(f'{import_name}.', '')

    huggingface_import = 'from transformers import PretrainedConfig, PreTrainedModel'
    delimiter = '$>'

    with open('./huggingface/huggingface_code.py', 'r') as f:
        contents = f.read()
        delim_location = contents.find(delimiter)
        cut_contents = contents[delim_location+len(delimiter):]

    with open(output_path, 'w') as f:
        f.write(f'{huggingface_import}\n{imports}\n{code}\n{cut_contents}')

def parse_args():
    parser = argparse.ArgumentParser(description="Compile huggingface wrapper")
    parser.add_argument("-o", type=str, help="Output filepath", default='./huggingface_wrapper.py')

    args = parser.parse_args()

    args.o = os.path.abspath(args.o)
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args.o)
