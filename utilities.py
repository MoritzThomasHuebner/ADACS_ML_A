# Utilities.py
# Dr. Matthew Smith, Swinburne University of Technology
# Various tools prepared for the ADACS Machine Learning workshop 

# Import modules
import matplotlib
import numpy
from scipy import signal

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np


def read_training_data(ID, N):
    # Will load files of names X_(Z).dat and Y_(Z).dat where
    # (Z) is an integer.
    # Example: Will load X_4.dat and Y_4.dat
    # X_4.dat will contain N double precision values.
    # Y_4.dat contains a single integer - the class (i.e. classification).
    # INPUT:
    # ID: This is the value of integer (Z) controlling which file to open
    # N: The number of elements contained within each time series
    # Files are in binary, hence precision needs to be provided.
    X = np.zeros(N)
    Y = np.zeros(1)
    fname = "LIGO_DATA/Train_Easy/X_LIGO_%d.dat" % ID
    print("Loading file " + fname)
    X = np.fromfile(fname, 'double')
    # Now for y
    fname = "LIGO_DATA/Train_Easy/Y_LIGO_%d.dat" % ID
    Y = np.fromfile(fname, 'double')
    return X, Y


def read_test_data(ID, N):
    # Will load files of names X_(Z).dat and Y_(Z).dat where
    # (Z) is an integer.
    # Example: Will load X_4.dat and Y_4.dat
    # X_4.dat will contain N double precision values.
    # Y_4.dat contains a single integer - the class (i.e. classification).
    # INPUT:
    # ID: This is the value of integer (Z) controlling which file to open
    # N: The number of elements contained within each time series
    # Files are in binary, hence precision needs to be provided.
    X = np.zeros(N);
    Y = np.zeros(1);
    fname = "LIGO_DATA/Test_Easy/X_LIGO_%d.dat" % ID
    print("Loading file " + fname)
    X = np.fromfile(fname, 'double')
    # Now for y
    fname = "LIGO_DATA/Test_Easy/Y_LIGO_%d.dat" % ID
    Y = np.fromfile(fname, 'double')
    return X, Y


def plot_results(ID, X):
    # Use Matplotlib to plot the data for inspection
    # ID: Data ID, only used for placing in the title.
    # X: Data we are plotting.
    fig, ax = plt.subplots()
    ax.plot(X)
    # Give it some labels
    Title = "Data Set %d" % ID
    ax.set(xlabel='Time Sequence (t)', ylabel='Data X(t)', title=Title)
    plt.show()
    return


def plot_history(history):
    # Use Matplotlib to view the convergence/training history
    fig, ax = plt.subplots()
    ax.plot(history.history['acc'])
    ax.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy Convergence History')
    plt.show()
    return

def swish(x):
    beta = 1.5
    return beta * x * np.exp(x) / (np.exp(x) + 1)


def normalize_data(data):
    return data / np.mean(np.abs(data))


def thin_data(data, thinning):
    return data[::thinning]


def filter_data(data, thinning):
    data = normalize_data(data)
    b, a = signal.butter(3, 0.01)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, data, zi=zi * data[0])
    data = signal.filtfilt(b, a, data)
    return thin_data(data, thinning)