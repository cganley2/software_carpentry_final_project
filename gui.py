import tkinter as tk
from tkinter import messagebox
import numpy as np
from import_model import import_model as im
import os


def gui():
    '''
    This function defines the looks and functionality of the GUI.
    '''

    main_window = tk.Tk()
    main_window.geometry('600x350')

    entry_width = 4

    x_start = 50
    x_label_start = 25
    x_interval = 80

    y_label_height = 220
    y_input_height = 250

    # TOP LABEL #
    top_label = tk.StringVar()
    top_label.set('Enter a number in each box between 0 and 1 (inclusive) that'
                  + '\n corresponds to the relative amount of each sensory'
                  + '\n input that you desire in your wine.')
    top_label = tk.Label(master=main_window, textvariable=top_label)
    top_label.place(x=x_label_start, y=100)

    # WINE TYPE #
    wine = tk.StringVar()
    red_type = tk.Radiobutton(master=main_window, text='Red', variable=wine,
                              value='red')
    red_type.place(x=x_start - 20, y=y_input_height)
    white_type = tk.Radiobutton(master=main_window, text='White',
                                variable=wine, value='white')
    white_type.place(x=x_start - 20, y=y_input_height + 20)

    wine_label = tk.StringVar()
    wine_label.set('Wine Type')
    wine_label = tk.Label(master=main_window, textvariable=wine_label)
    wine_label.place(x=x_label_start, y=y_label_height)

    # FIXED ACIDITY #
    fixed_acidity = tk.Entry(master=main_window, width=entry_width)
    fixed_acidity.pack()
    fixed_acidity.place(x=x_start + x_interval, y=y_input_height)

    FA_label = tk.StringVar()
    FA_label.set('Fixed Acidity')
    fixed_acidity_label = tk.Label(master=main_window, textvariable=FA_label)
    fixed_acidity_label.place(x=x_label_start + x_interval, y=y_label_height)

    # SOURNESS #
    sourness = tk.Entry(master=main_window, width=entry_width)
    sourness.pack()
    sourness.place(x=x_start + 2 * x_interval, y=y_input_height)

    sour_label = tk.StringVar()
    sour_label.set('Sourness')
    sour_label = tk.Label(master=main_window, textvariable=sour_label)
    sour_label.place(x=x_label_start + 2 * x_interval + 10, y=y_label_height)

    # SWEETNESS #
    sweetness = tk.Entry(master=main_window, width=entry_width)
    sweetness.pack()
    sweetness.place(x=x_start + 3 * x_interval, y=y_input_height)

    sweet_label = tk.StringVar()
    sweet_label.set('Sweetness')
    sweet_label = tk.Label(master=main_window, textvariable=sweet_label)
    sweet_label.place(x=x_label_start + 3 * x_interval, y=y_label_height)

    # ALCOHOL CONTENT #
    alcohol = tk.Entry(master=main_window, width=entry_width)
    alcohol.pack()
    alcohol.place(x=x_start + 4 * x_interval, y=y_input_height)

    alc_label = tk.StringVar()
    alc_label.set('Alcohol Content')
    alc_label = tk.Label(master=main_window, textvariable=alc_label)
    alc_label.place(x=x_label_start + 4 * x_interval, y=y_label_height)

    def predict_from_model():
        '''
        This function is called when the user clicks the Predict! button on
        the GUI. It validates input, loads the neural network model, then
        makes a prediction about the quality of the wine based on the inputs.
        '''

        sensory_dict = {'wine': True,
                        'fixed acidity': True,
                        'sourness': True,
                        'sweetness': True,
                        'alcohol': True}

        quality_dict = {'1': 'bad',
                        '2': 'good',
                        '3': 'great'}

        # VALIDATE INPUT #
        if wine.get() not in {'red', 'white'}:
            sensory_dict['wine'] = False
            raise ValueError('You must select a wine type (Red/White)')

        if fixed_acidity.get().isdigit():
            if not (0 <= float(fixed_acidity.get()) <= 1):
                sensory_dict['fixed acidity'] = False
                raise ValueError('Fixed acidity not in range [0, 1]')
        else:
            sensory_dict['fixed acidity'] = False
            raise TypeError('Fixed acidity must be a FLOAT in range [0, 1]')

        if sourness.get().isdigit():
            if not (0 <= float(sourness.get()) <= 1):
                sensory_dict['sourness'] = False
                raise ValueError('Sourness not in range [0, 1]')
        else:
            sensory_dict['sourness'] = False
            raise TypeError('Sourness must be a FLOAT in range [0, 1]')

        if sweetness.get().isdigit():
            if not (0 <= float(sweetness.get()) <= 1):
                sensory_dict['sweetness'] = False
                raise ValueError('Sweetness not in range [0, 1]')
        else:
            sensory_dict['sweetness'] = False
            raise TypeError('Sweetness must be a FLOAT in range [0, 1]')

        if alcohol.get().isdigit():
            if not (0 <= float(alcohol.get()) <= 1):
                sensory_dict['alcohol'] = False
                raise ValueError('Alcohol not in range [0, 1]')
        else:
            sensory_dict['alcohol'] = False
            raise TypeError('Alcohol must be a FLOAT in range [0, 1] \n\n'
                            + 'Illegal input type (str) ... \n\n'
                            + 'Alerting authorities ... \n\n'
                            + 'The police are on their way to your location.')
        # END VALIDATE INPUT #

        # load model and make prediction
        if all(value for value in sensory_dict.values()):
            model = im(wine.get())
            p = np.array([[fixed_acidity.get(),
                           sourness.get(),
                           sweetness.get(),
                           alcohol.get()]])

            prediction = model.predict(p)
            quality = str(list(prediction[0]).index(max(prediction[0])) + 1)
            messagebox.showinfo('RESULTS',
                                'This wine is {0}!'.format(quality_dict
                                                           [quality]))

    # PREDICT BUTTON #
    predict_button = tk.Button(master=main_window, text='Predict!',
                               command=predict_from_model)

    predict_button.pack()
    predict_button.place(x=530, y=y_input_height, height=30, width=60)

    main_window.mainloop()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gui()
