from keras.models import load_model


def import_model(wine_type):
    '''
    This function loads a trained neural network for use in the GUI. It only
    reads the 4 Columns Raw Data models because the GUI is intended to work
    with 4 'sensory' inputs that the user can specify.
    '''
    if wine_type == 'red':
        model_name = './neural_network_models/red_4 Columns Raw Data - Red.h5'
    else:
        model_name = './neural_network_models/white_4 Columns Raw Data - White.h5'

    model = load_model(model_name)

    return model
