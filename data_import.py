import pandas as pd

def data_import(filepath):
    wine_df = pd.read_csv(filepath, delimiter=';', header=0)
    # print(wind_df.head())
    

if __name__ == '__main__':
    filepath = './winequality-red.csv'
    data_import(filepath)
