### Core program for local denoising learning
### Works on Keras >= 1.4

import keras
from keras import backend as K
import os
import numpy as np

latent_dim = 39
n_features = 39

### Defining some useful functions here
### for simplicity, we use the linear mapping here
def network_encoder(x, latent_dim=latent_dim):
    x = keras.layers.Dense(units= 39, activation = 'linear', name = 'first_layer')(x)

    return x


## define the network integrates the information along the sequence
## zt- gar -> ct
### T_GRU in paper
def network_autoregressive(x):  ## to get Ct variable via RNN cell; GRU

  x = keras.layers.GRU(units = 39, activation='linear',
                       return_sequences=False, name = 'ar_context')(x)

  return x

## define mapping Ct -> other z_t+1, z_t+2
## T_L in papar
def network_prediction(context, latent_dim, predict_terms):

  outputs = []

  for i in range (predict_terms):
    outputs.append(keras.layers.Dense(units=latent_dim, activation="linear", name='z_t_{i}'.format(i=i))(context))
  if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
  else:
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

  return output



### Below is the model construction
encoder_input = keras.layers.Input(X_shape)
encoder_output = network_encoder(encoder_input, latent_dim)

encoder_model = keras.models.Model(encoder_input, encoder_output, name = 'encoder')

### Define the autoregressive part model

terms = 4  ## hyper-parameters for num of previous steps for producing Ct
predict_terms = 4

x_input = keras.layers.Input((terms, n_features)) ## x_t-3,...,x_t
x_encoded_sequence = keras.layers.TimeDistributed(encoder_model)(x_input) ## z_t-3,...,z_t

context = network_autoregressive(x_encoded_sequence) ## produce Ct variable

preds = network_prediction(context, latent_dim, predict_terms)  ## predict_terms: num of steps for prediction z_t+1,..., z_t+4 prediction based on previous values

y_input = keras.layers.Input((predict_terms, n_features )) ## x_t+1,..., x_t+4
y_encoded = keras.layers.TimeDistributed(encoder_model)(y_input) ## z_t+1,..., z_t+4; true values based on incoming input

### The D(X,C) network in paper; Mean == Expectation
dot_product = K.mean(y_encoded * preds, axis = -1)##
dot_product = K.mean(dot_product, axis = -1, keepdims = True)  ## avearge overall all prediction steps

dot_product_probs = K.sigmoid(dot_product)

loss = dot_product_probs

from keras.models import Model

Model = Model([x_input, y_input])
Model.add_loss(loss)
