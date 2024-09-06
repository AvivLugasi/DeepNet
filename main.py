import cupy as cp
import pandas as pd

from Functions.Activations.LeakyRelu import LeakyRelu
from Functions.Activations.Sigmoid import Sigmoid
from Functions.Activations.Tanh import Tanh
from Functions.Losses.BinaryCrossEntropy import BinaryCrossEntropy
from Functions.Losses.CrossEntropy import CrossEntropy
from Functions.Losses.MSE import MSE
from Functions.Metrics.Accuracy import Accuracy
from Initializers.GlorotHeInitializers import GlorotNormal, HeNormal, GlorotUniform, HeUniform
from Initializers.Zeroes import Zeroes
from Optimizers.SGD import SGD
from PreProcessing.Features_encoding import feature_one_hot
from PreProcessing.Normalization import standardization
from sklearn.model_selection import train_test_split
from Structures.Layers.Dense import Dense
from Structures.Layers.Dropout import Dropout, InvertedDropout
from Structures.Layers.Input import Input
from Structures.Layers.SoftMax import SoftMax
from Structures.Models.Model import Model

df_train = pd.read_csv('Datasets/Datasets/Mnist/Digits/mnist_train.csv')
df_test = pd.read_csv('Datasets/Datasets/Mnist/Digits/mnist_test.csv')
np_train = df_train.select_dtypes(include=[float, int]).values
np_test = df_test.select_dtypes(include=[float, int]).values

cp_train = cp.array(np_train, dtype=float)
cp_test = cp.array(np_test, dtype=float)

x_train, x_test, y_train, y_test = cp_train[:, 1:], cp_test[:, 1:], cp_train[:, 0], cp_test[:, 0]
x_train = standardization(x_train, axis=0)
x_test = standardization(x_test, axis=0)

x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.reshape(-1, 1).T, y_test.reshape(-1, 1).T
y_train, y_test = y_train.astype(cp.int64), y_test.astype(cp.int64)
y_train = feature_one_hot(mat=y_train, feature_row=0)
y_test = feature_one_hot(mat=y_test, feature_row=0)

input_l = Input(features_are_rows=True)
dense_1 = Dense(units=45,
                activation=LeakyRelu(),
                use_bias=True,
                weights_init_method=HeUniform(),
                bias_init_method=Zeroes(),
                weights_regularizer=None,
                xp_module=cp)
dense_2 = Dense(units=45,
                activation=LeakyRelu(),
                use_bias=True,
                weights_init_method=HeUniform(),
                bias_init_method=Zeroes(),
                weights_regularizer=None,
                xp_module=cp)
dense_3 = Dense(units=10,
                activation=LeakyRelu(),
                use_bias=True,
                weights_init_method=HeUniform(),
                bias_init_method=Zeroes(),
                weights_regularizer=None,
                xp_module=cp)
softmax = SoftMax()
dropout_1 = InvertedDropout(keep_prob=0.8)
dropout_2 = InvertedDropout(keep_prob=0.6)

m = Model(input_layer=input_l, hidden_layers=[dense_1, dropout_1, dense_2, dense_3, softmax])
m.compile(optimizer=SGD(init_learning_rate=0.01, momentum=0.9), loss=CrossEntropy(), metrics=[Accuracy()])
m.fit(y_train=y_train,
      x_train=x_train,
      epochs=1500,
      batch_size=512,
      validation_data=(x_test, y_test),
      shuffle=True)

m.evaluate(x_test=x_test, y_test=y_test, samples_as_cols=True)
