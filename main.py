import cupy as cp
import pandas as pd

from Functions.Activations.LeakyRelu import LeakyRelu
from Functions.Activations.Sigmoid import Sigmoid
from Functions.Activations.Tanh import Tanh
from Functions.Losses.BinaryCrossEntropy import BinaryCrossEntropy
from Functions.Losses.CrossEntropy import CrossEntropy
from Functions.Losses.MSE import MSE
from Functions.Metrics.Accuracy import Accuracy
from Initializers.GlorotHeInitializers import GlorotNormal, HeNormal, GlorotUniform
from Initializers.Zeroes import Zeroes
from Optimizers.SGD import SGD
from PreProcessing.Normalization import standardization
from sklearn.model_selection import train_test_split
from Structures.Layers.Dense import Dense
from Structures.Layers.Dropout import Dropout, InvertedDropout
from Structures.Layers.Input import Input
from Structures.Models.Model import Model

df = pd.read_csv('Datasets/Datasets/HeartDiseasePredictions/heart-disease.csv')

# Convert the DataFrame to a NumPy ndarray (excluding non-numeric columns if necessary)
numpy_array = df.select_dtypes(include=[float, int]).values

# Convert the NumPy ndarray to a CuPy ndarray
cupy_array = cp.asarray(numpy_array)

X = cupy_array[:, :-1]  # All rows, all columns except the last

x_normalized = standardization(X, axis=0)

y = cupy_array[:, -1].reshape(-1, 1)   # All rows, last column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


X_train, X_test = X_train.T, X_test.T
y_train, y_test = y_train.T, y_test.T


input_l = Input(features_are_rows=True)
dense_1 = Dense(units=5,
                activation=Tanh(),
                use_bias=True,
                weights_init_method=GlorotUniform(),
                bias_init_method=Zeroes(),
                weights_regularizer=None,
                xp_module=cp)
dense_2 = Dense(units=5,
                activation=Tanh(),
                use_bias=True,
                weights_init_method=GlorotUniform(),
                bias_init_method=Zeroes(),
                weights_regularizer=None,
                xp_module=cp)
dense_3 = Dense(units=10,
                activation=Tanh(),
                use_bias=True,
                weights_init_method=GlorotUniform(),
                bias_init_method=Zeroes(),
                weights_regularizer=None,
                xp_module=cp)
dense_4 = Dense(units=1,
                activation=Sigmoid(),
                use_bias=True,
                weights_init_method=GlorotUniform(),
                bias_init_method=Zeroes(),
                weights_regularizer=None,
                xp_module=cp)
dropout_1 = InvertedDropout(keep_prob=0.8)
dropout_2 = InvertedDropout(keep_prob=0.6)

m = Model(input_layer=input_l, hidden_layers=[dense_1, dense_2, dense_4])
m.compile(optimizer=SGD(init_learning_rate=0.01), loss=BinaryCrossEntropy(), metrics=[Accuracy()])
m.fit(y_train=y_train,
      x_train=X_train,
      epochs=600,
      batch_size=272,
      validation_data=(X_test, y_test),
      shuffle=True)

m.evaluate(x_test=X_test, y_test=y_test, samples_as_cols=True)
