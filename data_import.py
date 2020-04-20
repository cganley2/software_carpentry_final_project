import pandas as pd


def data_import(filepath):
    wine_df = pd.read_csv(filepath, delimiter=';', header=0)
    return wine_df


if __name__ == '__main__':
    filepath = './winequality-red.csv'
    df = data_import(filepath)
    print(df.head())
