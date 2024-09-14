import cupy as cp
import pandas as pd

from Functions.Activations.Elu import Elu
from Functions.Activations.LeakyRelu import LeakyRelu
from Functions.Activations.Relu import Relu
from Functions.Activations.Sigmoid import Sigmoid
from Functions.Activations.Tanh import Tanh
from Functions.Losses.BinaryCrossEntropy import BinaryCrossEntropy
from Functions.Losses.CrossEntropy import CrossEntropy
from Functions.Losses.MSE import MSE
from Functions.Metrics.Accuracy import Accuracy
from Initializers.GlorotHeInitializers import GlorotNormal, HeNormal, GlorotUniform, HeUniform
from Initializers.Zeroes import Zeroes
from Optimizers.Adagrad import Adagrad
from Optimizers.Adam import Adam
from Optimizers.RMSprop import RMSprop
from Optimizers.SGD import SGD
from Optimizers.Schedules.CosineDecay import CosineDecay
from Optimizers.Schedules.ExponentialDecay import ExponentialDecay
from Optimizers.Schedules.InverseTimeDecay import InverseTimeDecay
from Optimizers.Schedules.PiecewiseConstantDecay import PiecewiseConstantDecay
from PreProcessing.Features_encoding import feature_one_hot
from PreProcessing.Normalization import standardization, EPSILON
from sklearn.model_selection import train_test_split

from Regularization.L1 import L1
from Regularization.L1L2 import L1L2
from Regularization.L2 import L2
from Structures.Layers.BatchNorm import BatchNorm
from Structures.Layers.Dense import Dense
from Structures.Layers.Dropout import Dropout, InvertedDropout
from Structures.Layers.Input import Input
from Structures.Layers.SoftMax import SoftMax
from Structures.Models.Model import Model

# digit mnist
df_train = pd.read_csv('Datasets/Datasets/Mnist/Digits/mnist_train.csv')
df_test = pd.read_csv('Datasets/Datasets/Mnist/Digits/mnist_test.csv')
# fashion mnist
# df_train = pd.read_csv('Datasets/Datasets/Mnist/Fashion/fashion-mnist_train.csv')
# df_test = pd.read_csv('Datasets/Datasets/Mnist/Fashion/fashion-mnist_test.csv')
# letters mnist
# df_train = pd.read_csv('Datasets/Datasets/Mnist/EMnist/emnist-letters-train.csv')
# df_test = pd.read_csv('Datasets/Datasets/Mnist/EMnist/emnist-letters-test.csv')
np_train = df_train.select_dtypes(include=[float, int]).values
np_test = df_test.select_dtypes(include=[float, int]).values

cp_train = cp.array(np_train, dtype=float)
cp_test = cp.array(np_test, dtype=float)

x_train, x_test, y_train, y_test = cp_train[:, 1:], cp_test[:, 1:], cp_train[:, 0], cp_test[:, 0]
x_train, mean_train, std_train = standardization(x_train, axis=0, return_params=True)
x_test = (x_test - mean_train) / (cp.sqrt(std_train**2 + EPSILON))

x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.reshape(-1, 1).T, y_test.reshape(-1, 1).T
y_train, y_test = y_train.astype(cp.int64), y_test.astype(cp.int64)
y_train = feature_one_hot(mat=y_train, feature_row=0)
y_test = feature_one_hot(mat=y_test, feature_row=0)
print(y_train.shape)
input_l = Input(features_are_rows=True)
dense_1 = Dense(units=30,
                activation=LeakyRelu(alpha_value=0.05),
                use_bias=False,
                weights_init_method=HeNormal(),
                bias_init_method=Zeroes(),
                xp_module=cp,
                weights_regularizer=L2(l2=0.01),
                batchnorm=BatchNorm(vectors_size=30))
dense_2 = Dense(units=30,
                activation=LeakyRelu(alpha_value=0.05),
                use_bias=False,
                weights_init_method=HeNormal(),
                bias_init_method=Zeroes(),
                xp_module=cp,
                weights_regularizer=L2(l2=0.01),
                batchnorm=BatchNorm(vectors_size=30))
dense_3 = Dense(units=10,
                activation=Elu(),
                use_bias=False,
                weights_init_method=HeNormal(),
                bias_init_method=Zeroes(),
                xp_module=cp,
                weights_regularizer=L2(l2=0.01),
                batchnorm=BatchNorm(vectors_size=10))
softmax = SoftMax()
dropout_1 = InvertedDropout(keep_prob=0.75)
dropout_2 = InvertedDropout(keep_prob=0.6)

# pwc_d = PiecewiseConstantDecay(boundaries=[5000, 10000, 15000, 20000],
#                                values=[0.15, 0.1, 0.05, 0.01])
exp_d = ExponentialDecay(learning_rate=0.13,
                         decay_steps=8000,
                         decay_rate=0.95)
sgd = SGD(momentum=0.9, init_learning_rate=exp_d)
rmsprop = RMSprop(init_learning_rate=exp_d,
                  optimizer_momentum=0.90,
                  momentum=0.9)
adgrad = Adagrad(init_learning_rate=0.95,
                 momentum=0.9)
adam = Adam(init_learning_rate=0.15,
            momentum=0.9,
            optimizer_momentum=0.999)
m = Model(input_layer=input_l, hidden_layers=[dense_1, dense_2, dense_3, SoftMax()])
m.compile(optimizer=adam, loss=CrossEntropy(), metrics=[Accuracy()])
m.fit(y_train=y_train,
      x_train=x_train,
      epochs=50,
      batch_size=512,
      validation_split=0.2,
      shuffle=True)

m.evaluate(x_test=x_test, y_test=y_test, samples_as_cols=True)
