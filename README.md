# Prediction of Hourly Energy Consumption using RNN and LSTM

### Introduction:
In recent years, advances in sensor technologies and expansion of smart meters have resulted in massive growth of energy data sets. These big data have created new opportunities for energy prediction, but at the same time, they impose new challenges for traditional technologies. That is the reason I have attempted to solve this problem of prediction using deep learning, like recurrent neural networks.
Prediction of energy consumption is important for the following reasons:
1.	Saving energy just in case it is depleted in the future
2.	Estimation of the quantity of fuel required to produce energy
3.	Planning for the future
A Recurrent Neural Network deals with sequence problems because their connections form a directed cycle. They can retain state from one iteration to the next by using their own output as input for the next step. The hourly energy consumption data has a sequential pattern, because a particular hour’s data depends on its previous hour’s data. Thus, RNNs are used for this prediction.

### Datasets:
I have used three different datasets to test the models. Those datasets are; The Dayton Power and Light Company, American Electric Power (AEP) and PJM Load. This hourly power consumption data is taken from Kaggle.
The datasets have the datetime in the first column and the second column has the energy consumption in megawatts (MW) range. I have split the data into training and testing data and used only the training examples for training the model. This division is given in the subsections for each dataset below.

### Data Pre-processing:
The sklearn package MinMaxScaler is used for normalization of the data before using it. I converted the data into the range -1 to 1. Then, the data is divided into training and testing sets where maximum possible data is taken for training. The data is properly shaped into a matrix form.

### RNN:
Recurrent Neural Networks are artificial neural networks where connections between nodes form a directed graph along a temporal sequence. Each hidden layer is provided with an input and the output from the previous layer. RNNs can use their internal state (memory) to process sequences of inputs. This makes them suitable for time series data like hourly energy consumption.
Vanishing gradient problem:
During the gradient descent using back-propagation, to update weights, the gradients become so small that the weights are no more updated. This is the vanishing gradient problem. In RNNs, the gradient of the loss function decays exponentially with time (called the vanishing gradient problem). This hinders them from using long term information. They are good for storing memory 3-4 instances of past iterations but larger number of instances don't provide good results.

### LSTM:
LSTM networks are a type of RNN that uses special units in addition to standard units. LSTM units include a 'memory cell' that can maintain information in memory for long periods of time. A set of gates is used to control when information enters the memory, when it is output, and when it is forgotten. This architecture lets them learn longer-term dependencies. Thus, LSTMs overcome the vanishing gradient problem. A single LSTM unit is as shown below:

 
