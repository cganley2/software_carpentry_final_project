from data_import import data_import as di
from pc_analysis import pc_analysis


def run():
    filepath_red = './winequality-red.csv'
    filepath_white = './winequality-white.csv'
    wine_df = di(filepath_red)
    pc_analysis(wine_df, plot=False)
    # print(min(wine_df['quality']))
    # print(max(wine_df['quality']))

if __name__ == '__main__':
    run()
