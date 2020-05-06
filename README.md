# software_carpentry_final_project
Final project for EN 540.635 Software Carpentry
Due 06 May 2020
Connor Ganley

This project is comprised of 3 main parts: principal component analysis (PCA)
on a data set of 11 features and 1 target, training neural networks based on
PCA data and raw data, and a graphical user interface (GUI) that uses a trained
neural network to predict quality based on 4 data points.

<b>Dataset</b>

The dataset used in this project contains \~1e3 data points of 11 features and
1 target dealing with measurements associated with two types of wine: red and
white. The target is the "quality" of the wine, rated on a 1-10 scale, with 10
being the best. The data can be found at:
https://archive.ics.uci.edu/ml/datasets/Wine+Quality

<b>PCA</b>

The PCA is conducted within the pc_analysis file. For N = 1 up to 11 principal
components, PCA is done such that the program reduces the 11 features to N
dimensions. The performance of PCA is shown in the /neural_network_performance/
directory. The performance was measured by how much variance the PCA captured,
and it is shown in the row labeled "Variance captured in PCA (%):". Because
this data set was fairly small (only ~1e3 data points), the PCA for smaller N
did not capture an acceptable amount of variance, so it was not particularly
useful in later calculations, apart from a comprising a holistic look at PCA
and its relevance to neural networks. This is to say that PCA was only useful
when the 11 features were reduced to ~8 dimensions, which is not that much of
a dimensionality reduction, typically the purpose of performing PCA.

<b>Neural Networks</b>

The training of neural networks is conducted within the nn file. It trains a
10-8-6-6-4-2 neural network on each PCA reduction as well as on all the raw
data for each wine type, and finally on a 4-column simplification of raw data
corresponding to sensory data for use in the GUI. It is important to mention
here that a classification reduction was necessary, believed to be the result
of the size of the data set. That is, classifying wine data into 10 bins
(1-10 scale) was not very accurate. During initial attempts to do so, only
40-50% validation accuracy was achieved, which is little better than guessing.
I believe this could be made more feasible with at least a 10-fold increase in
the number of training data points. With that said, to make training meaningful,
the scale was reduced to three possible classifications: "great," "good," and
"bad." Wine quality >= 7 was considered "great,"" wine quality < 4 was "bad,"
with the remainder being "good." This led to a neural network validation
accuracy of greater than 80%, which was acceptable for the scope of this
project. In the training of the neural networks (26 in total), a training set
comprised of 80% of the data with the balance as the test set was used. The
performance of each neural network can be seen in the
/neural_network_performance/ directory, where the status of the network after
10 training epochs is shown. The neural network models are stored in the
/neural_network_models/ directory.

<b>GUI</b>

The GUI is created by the gui file. It creates a graphical user interface in
which the user can enter 4 values between 0 and 1 corresponding to the amount
of each sensory input listed. The user must also select a wine type: red or
white. When the user clicks the "Predict!" button, the neural network model
trained on those 4 features predicts the quality of a wine with those four
attributes and shows the user the result in a new window.

<b>Using this program</b>

To conduct a new batch of PCA and neural network trainings, the user should
open and run the "generate_models.py" module. The new output data will be
stored in directories similarly named to the \_performance and \_models ones
currently in the repository.

To review previously generated models and PCA, the user should peruse the files
present in /neural_network_performance/ and /neural_network_models/ directories.
These were included to demonstrate that the modules worked and produced
meaningful output upon submission of this project.

To use the GUI, the user should open and run the "gui.py" module. From there,
they can enter values and predict to their heart's content.