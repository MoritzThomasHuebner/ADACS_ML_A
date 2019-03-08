# train.py
# Written by Dr. Matthew Smith, Swinburne University of Technology
# Prepared as training material for ADACS Machine Learning workshop
# This is an example of time series (sequence) classification
# for a binary classification problen.

# Import modules
from keras.models import Sequential
from keras.layers import Dense, Dropout
# from keras.layers import LSTM
from utilities import *

# Create training arrays
# In this demonstration I create our numpy arrays and then
# load each time sequence in one-by-one.
from utilities import filter_data

N_train = 400  # Number of elements to train
N_test = 50  # Number of elements to train
N_sequence = 200000  # Length of each piece of data
thinning = 10
N_sequence_thinned = int(N_sequence / thinning)
N_epochs = 100  # Number of epochs

# Create the training sequence data (X) and each set's classification (Y).
X_train = np.empty([N_train, N_sequence])
Y_train = np.empty(N_train)

# Load the data from file
for x in range(N_train):
    # This will create x = 0, 1, 2...to N_train-1
    X_train[x,], Y_train[x] = read_training_data(x + 1, N_sequence)

X_train_pp = np.empty([N_train, int(N_sequence_thinned)])
X_test_pp = np.empty([N_test, int(N_sequence_thinned)])

# Also create the numpy arrays for the testing data set
X_test = np.empty([N_test, N_sequence])
Y_test = np.empty(N_test)
for x in range(N_train):
    X_train[x, ], Y_train[x] = read_training_data(x + 1, N_sequence)
    X_train_pp[x, ] = filter_data(X_train[x,], thinning)
    # if x % 50 == 0:
    #     plot_results(Y_train[x], np.fft.fft(X_train_pp[x]))
    #     plot_results(Y_train[x], X_train_pp[x])

for x in range(N_test):
    X_test[x, ], Y_test[x] = read_test_data(x + 1, N_sequence)
    X_test_pp[x, ] = filter_data(X_test[x, ], thinning)
    # if x % 50 == 0:
    #     plot_results(Y_train[x], X_test_pp[x])

# print(Y_test)
# print(Y_train)
model = Sequential()

# Configure our RNN by adding neural layers with activation functions
# model.add(Dropout(0.3, input_shape=(N_sequence_thinned,)))
model.add(Dropout(0.2, input_shape=(N_sequence_thinned,)))
model.add(Dense(16, activation='relu', input_dim=N_sequence_thinned))
model.add(Dense(8, activation='softmax', input_dim=N_sequence_thinned))
model.add(Dense(1, activation='sigmoid', input_dim=N_sequence_thinned))
# model.add(Dense(128, activation='relu', input_dim=N_sequence_thinned))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='softmax'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation=swish))
# model.add(Dense(16, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(8, activation='tanh'))
# model.add(Dense(8, activation='tanh'))
# model.add(Dropout(0.4))
# model.add(Dense(1, activation='sigmoid'))

# The commented lines below are designed to offer some insight
# into the use of layers without activation functions.
# The 3 lines above should be commented out before uncommenting these.
# No activation layers
# model.add(Dense(16,input_dim=N_sequence))
# model.add(Dense(1))

# Compile model and print summary
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# Fit the model using the training set
history = model.fit(X_train_pp, Y_train, epochs=N_epochs, batch_size=32)

# Plot the history
plot_history(history)

# Final evaluation of the model using the Test Data
print("Evaluating Test Set")
scores = model.evaluate(X_test_pp, Y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# Export the model to file
model_json = model.to_json()
with open("model.json", "w") as json_file:
        json_file.write(model_json)
# Save the weights as well, as a HDF5 format
model.save_weights("model.h5")
