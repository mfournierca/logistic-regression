import pandas as pd
from os import path


def parse_data(data_root, data_file="spambase.data"):
    df = pd.read_csv(
        path.join(data_root, data_file), 
        header=None, 
        names=header_names(data_root)
    )
    return df


def header_names(data_root, names_file="spambase.names"):
    names = []
    with open(path.join(data_root, names_file), "r") as f:
        for i, e in enumerate(f):
            if 32 < i < 90:
                names.append(e[:e.index(":")])
    names.append("spam")
    return names
