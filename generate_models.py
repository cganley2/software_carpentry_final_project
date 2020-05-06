from data_import import data_import as di
from pc_analysis import pc_analysis
from nn import nn
from save_output import save_output
import os
import pandas as pd


def generate_models():
    '''
    This function generates 26 neural network models; 13 for each type of wine:
    red and white. 11 of those models utilize PCA data, 1 is trained on all raw
    feature data, and 1 is trained on 4 sensory features to equal 13 per type.
    '''

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    filepath_red = './raw_data/winequality-red.csv'
    filepath_white = './raw_data/winequality-white.csv'

    red_wine_df = di(filepath_red)
    red_pc_dict = pc_analysis(red_wine_df)
    red_pc_model_list = nn(red_pc_dict)

    red_raw_quality = pd.DataFrame(red_wine_df['quality'])
    red_raw_quality.columns = ['quality']
    del red_wine_df['quality']

    red_raw_data_dict = {'11 Raw Data - Red':
                         [None, red_wine_df, red_raw_quality]}

    red_raw_data_model = nn(red_raw_data_dict)

    red_4_columns = pd.DataFrame((red_wine_df['fixed acidity'],
                                  red_wine_df['citric acid'],
                                  red_wine_df['residual sugar'],
                                  red_wine_df['alcohol'])).transpose()

    red_4_columns_dict = {'4 Columns Raw Data - Red':
                          [None, red_4_columns, red_raw_quality]}

    red_4_columns_model = nn(red_4_columns_dict)

    red_model_list = (red_pc_model_list
                      + red_raw_data_model
                      + red_4_columns_model)

    save_output(red_model_list, 'red_performance.txt', red_pc_dict)

    white_wine_df = di(filepath_white)
    white_pc_dict = pc_analysis(white_wine_df)
    white_pc_model_list = nn(white_pc_dict)

    white_raw_quality = pd.DataFrame(white_wine_df['quality'])
    white_raw_quality.columns = ['quality']
    del white_wine_df['quality']

    white_raw_data_dict = {'11 Raw Data - White':
                           [None, white_wine_df, white_raw_quality]}

    white_raw_data_model = nn(white_raw_data_dict)

    white_4_columns = pd.DataFrame((white_wine_df['fixed acidity'],
                                    white_wine_df['citric acid'],
                                    white_wine_df['residual sugar'],
                                    white_wine_df['alcohol'])).transpose()

    white_4_columns_dict = {'4 Columns Raw Data - White':
                            [None, white_4_columns, white_raw_quality]}

    white_4_columns_model = nn(white_4_columns_dict)

    white_model_list = (white_pc_model_list
                        + white_raw_data_model
                        + white_4_columns_model)

    save_output(white_model_list, 'white_performance.txt', white_pc_dict)


if __name__ == '__main__':
    generate_models()
