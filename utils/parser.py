import argparse as ag
import json
def get_parser_with_args(metadata_json='metadata.json'):#这里我修改啦！——hyy
    parser = ag.ArgumentParser(description='Training change detection network')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)
        return parser, metadata

    return None
