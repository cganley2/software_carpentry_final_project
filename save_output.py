import os


def save_output(model_list, filename, input_dict=None):
    '''
    This function saves the output of the 26 neural networks trained by
    generate_models.py to a text file such that the user can compare their
    performances. It then saves the trained neural networks to .h5 files in
    case the user wants to test values on them.

    ** Parameters **

        model_list: *list, nn_output objects*
            list of neural network models and histories, performance data
        filename: *str*
            name of file to which the performance data will be written
        input_dict: *dictionary* {'# Principal Components': [PCA object,
                                                             features_df,
                                                             target_df]}
            dictionary of PCA object data (used for variance), df of PCs, and
            df of wine quality. PCA object data will be None if training on
            non-PCA dataset

    ** Returns **

        None
    '''

    if 'red' in filename:
        prefix = 'red_'
    elif 'white' in filename:
        prefix = 'white_'

    top_dir = os.getcwd()

    nn_performance = 'github_user_neural_network_performance'
    if not os.path.exists(nn_performance):
        os.mkdir(nn_performance)
        os.chdir(top_dir + '/' + nn_performance)
    else:
        os.chdir(top_dir + '/' + nn_performance)

    with open(filename, 'w') as f:
        for model in model_list:
            f.write(str(model.name) + '\n')
            f.write('loss: ' + str(model.loss[-1]) + ' - ')
            f.write('accuracy: ' + str(model.accuracy[-1]) + ' - ')
            f.write('val_loss: ' + str(model.val_loss[-1]) + ' - ')
            f.write('val_accuracy: ' + str(model.val_accuracy[-1]) + '\n')
            if model.name in input_dict.keys():
                f.write('Variance captured in PCA (%): '
                        + str(sum(input_dict[model.name]
                                  [0].explained_variance_ratio_) * 100)
                        + '\n\n')
            else:
                f.write('\n')
    f.close()

    nn_models = 'github_user_neural_network_models'
    os.chdir(top_dir)

    if not os.path.exists(nn_models):
        os.mkdir(nn_models)
        os.chdir(top_dir + '/' + nn_models)
    else:
        os.chdir(top_dir + '/' + nn_models)
    save_dir = os.getcwd()

    for model in model_list:
        model_path = os.path.join(save_dir, str(prefix + model.name + '.h5'))
        model.model.save(model_path)

    os.chdir(top_dir)
