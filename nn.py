from data_import import data_import as di
from pc_analysis import pc_analysis
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.normalization import BatchNormalization


class nn_output:
    '''
    This is a class for storing neural network output data.
    '''

    def __init__(self, name, model, history):
        self.name = name
        self.val_loss = history.history['val_loss']
        self.val_accuracy = history.history['val_accuracy']
        self.loss = history.history['loss']
        self.accuracy = history.history['accuracy']
        self.model = model


def nn(input_dict):
    '''
    This function trains a neural network from the data given to it in
    input_dict. This code was based on that provided in Lab8.py as created by
    Dr. Henry Herbol for his EN 540.635 Software Carpentry course, through
    instructors Isaiah Chen and Divya Sharma.

    ** Parameters **

        input_dict: *dictionary* {'# Principal Components': [PCA object,
                                                             PC_df,
                                                             target_df]}
            dictionary of PCA data, feature(s) data, target data

    ** Returns **

        model_list: *list, nn_output objects*
            list of neural network models and histories, performance data
    '''

    model_list = []

    # input_dict = {'# PC': [PCA, final_df_x, final_df_y]} for reference
    for key, value in input_dict.items():

        print('\n' + key + '\n')

        # normalize x data BUT NOT Y DATA
        for col in value[1]:
            x = value[1].values
            min_max_scaler = preprocessing.MinMaxScaler()
            normalized_x_data = pd.DataFrame(min_max_scaler.fit_transform(x),
                                             columns=value[1].columns)

        # wine quality binning:
        # 3 : great
        # 2 : good
        # 1 : bad
        value[2] = value[2].apply(lambda x: [3 if val >= 7
                                             else 1 if val < 4
                                             else 2 for val in x])

        all_data = pd.concat([normalized_x_data, value[2]], axis=1)

        # demarcate normalized training (80%) and test sets (20%)
        train, test = train_test_split(all_data, test_size=0.2,
                                       stratify=value[2])

        train = train.astype('float32')
        test = test.astype('float32')

        y_train = train['quality']
        del train['quality']
        x_train = train

        y_test = test['quality']
        del test['quality']
        x_test = test

        n_classes = 3
        y_train = np_utils.to_categorical(y_train - 1, n_classes)
        y_test = np_utils.to_categorical(y_test - 1, n_classes)

        model, history = build_network(x_train, y_train,
                                       x_test, y_test, int(key[:2]))

        # store model, history in a class
        model_list.append(nn_output(key, model, history))

    return model_list


def build_network(x_train, y_train, x_test, y_test, components):
    '''
    This function holds the neural network model that is trained and tested
    with the wine data preprocessed in nn() above.

    The structure of the neural network was based on that shown in this video:
    https://www.youtube.com/watch?v=YWDdACh9EXk by John G. Fisher

    **Parameters**

        x_train: *numpy.ndarray*
            The set of training data
        y_train: *numpy.ndarray*
            The set of labels for the corresponding training data
        x_test: *numpy.ndarray*
            The set of testing data
        y_test: *numpy.ndarray*
            The set of labels for the corresponding testing data
        components: *int*
            The integer number of features used in the input structure

    **Returns**

        model:
            A class object that holds the trained model.
        history:
            A class object that holds the history of the model.

    '''
    # Build the model
    relu = 'relu'

    model = Sequential()

    # input layer
    model.add(Dense(10, input_shape=(components,)))
    model.add(Activation(relu))

    # hidden layer
    model.add(Dense(8))
    model.add(Activation(relu))

    model.add(Dense(6))
    model.add(Activation(relu))

    model.add(Dense(6))
    model.add(Activation(relu))

    model.add(Dense(4))
    model.add(Activation(relu))

    model.add(Dense(2))
    model.add(Activation(relu))

    # output layer
    model.add(Dense(3))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train,
                        batch_size=32,
                        epochs=10,
                        verbose=0,
                        validation_data=(x_test, y_test))

    # Return the model and history
    return model, history


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    filepath_red = './winequality-red.csv'
    filepath_white = './winequality-white.csv'
    wine_df = di(filepath_red)

    pc_dict = pc_analysis(wine_df, plot=False)

    # test_pc_dict = {'11 PC': [1, wine_df, quality]}
    # test_pc_dict = {'1 PC': pc_dict['1 PC']}

    model_list = nn(pc_dict)

    quality = pd.DataFrame(wine_df['quality'])
    quality.columns = ['quality']
    del wine_df['quality']
    unadulterated_dict = {'11 raw data': [1, wine_df, quality]}

    # model_list = nn(unadulterated_dict)
