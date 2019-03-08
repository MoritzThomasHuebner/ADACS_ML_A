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
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
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


def swish(x):
    # The swish activation function
    # Not currently used, but you *might* want to.
    beta = 1.5
    return beta * x * np.exp(x) / (np.exp(x) + 1)


def Preview_Image_Generator(flip_mode):
    #  This function is designed to give us a feel for the behaviour
    # of the imagedatagenerator function shipped with Keras.
    # It will augment the data contained in 0.jpg (cats, 'cause cats are cool)
    # and produce variations of this image, saved in ./preview
    datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=flip_mode, fill_mode='nearest')

    img = load_img('./cats_dogs/train/cats/0.jpg')  # Work with image 0 to start with
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely

    return 0


def Prepare_Image_Data(shear, zoom, flip_mode, test_flag):
    # Decide what features to use in ImageDataGenerator depending on
    # whether or not we are training or testing.
    if (test_flag == False):
        # Training data - will employ data augmentation, so can flip, zoom, shear etc.
        # Normalise the RGB data to between 0 and 1
        datagen = ImageDataGenerator(rescale=1.0 / 255.0,
                                     shear_range=shear, zoom_range=zoom, horizontal_flip=flip_mode)
    else:
        # We don't need to augment the test data set - no reason to shift, zoom or flip.
        # Still need to rescale, though, since training employed normalisation.
        datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    return datagen
