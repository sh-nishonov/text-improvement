import pandas as pd

def load_terms():
    return pd.read_csv("./data/terms.csv")['terms'].to_list()