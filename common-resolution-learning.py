### Common-resolution learning module


from keras import  Model
from keras.layers import  Input,Dense,concatenate,Add
from keras import backend as K


### Zero padding functions for cross generation later on
from keras.engine.topology import Layer

class ZeroPadding(Layer):
     def __init__(self, **kwargs):
          super(ZeroPadding, self).__init__(**kwargs)
     def call(self, x, mask=None):
          return K.zeros_like(x)

     def get_output_shape_for(self, input_shape):
          return input_shape


### Correlation loss function (cosine similarity between x1 and x2, and et ac)
def correlationLoss(fake,H):
  y1 = H[:,:39]
  y2 = H[:,39:]
  y1_mean = K.mean(y1, axis=0)
  y1_centered = y1 - y1_mean
  y2_mean = K.mean(y2, axis=0)
  y2_centered = y2 - y2_mean
  corr_nr = K.sum(y1_centered * y2_centered, axis=0)
  corr_dr1 = K.sqrt(K.sum(y1_centered * y1_centered, axis=0) + 1e-8)
  corr_dr2 = K.sqrt(K.sum(y2_centered * y2_centered, axis=0) + 1e-8)
  corr_dr = corr_dr1 * corr_dr2
  corr = corr_nr / corr_dr
  return K.sum(corr) * 0.01


## The case of 39 ROI time series case
X_input = Input(shape = (39,))
Y_input = Input(shape = (39,))

hl = Dense(39, activation='linear')(X_input)

hr = Dense(39, activation = 'linear')(Y_input)

h = Add()([hl, hr])

#decoder

recX = Dense(39, activation='linear')(h)
recY = Dense(39, activation='linear')(h)

CorrNet = Model([X_input, Y_input], [recX, recY, h])

### Three loss terms in paper: L_self;L_cross;L_cor
[recx0,recy0,h0] = CorrNet( [X_input, Y_input])  ## L(zi, g(h(z)))
[recx1,recy1,h1] = CorrNet( [X_input, ZeroPadding()(Y_input)]) #L(z, g(h(x)))
[recx2,recy2,h2] = CorrNet( [ZeroPadding()(X_input), Y_input]) # L(z,g(h(y)))
H= concatenate([h1,h2]) ##lambda* corr(h(x), h(Y))
model = Model( [X_input,Y_input],[recx0,recx1,recx2,recy0,recy1,recy2,H])

### Simple MSE loss is used here, diff loss can be considered
model.compile(loss = ['mse', 'mse', 'mse',
                     'mse', 'mse', 'mse',
                     correlationLoss], optimizer='adam')
