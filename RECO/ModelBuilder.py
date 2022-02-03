import keras.backend as K
from keras.constraints import Constraint
from keras.models import Model
from keras.layers import Dense, Activation, Input, Layer, BatchNormalization            



class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range'''
    
    def __init__(self, c=2):
        self.c = c
        
    def __call__(self, p):
        return K.clip(p, -self.c, self.c)
    
    def get_config(self):
        return {'name': self.__class__.__name__, 'c': self.c}


class ModelBuilder:
    '''Builds the NN architecture'''
    
    def __init__(self, 
                 n_input: int, 
                 latentsize: int = 3, layers: int = 1, weight_clipping: float = 7., 
                 internal_activation: str = 'tanh',
                 batch_norm_bool: bool = False,
                 more_batch_norm_bool: bool = False,
                 custom_activation_bool: bool = False,
                 custom_const: float = 1.
                ):
        '''initialize NN parameters and user choices'''
        
        self.n_input = n_input
        self.latentsize = latentsize
        self.layers = layers
        self.weight_clipping = weight_clipping
        self.internal_activation = internal_activation
        self.batch_norm_bool = batch_norm_bool
        self.more_batch_norm_bool = more_batch_norm_bool
        self.custom_activation_bool = custom_activation_bool
        self.custom_const = custom_const
    
    
    def __call__(self):
        '''call method: builds the NN'''
        
        def custom_activation(x):
            '''custom activation function for the output layer'''
            return self.custom_const*((K.sigmoid(x/self.custom_const) * 2) - 1) 
        
        inputs = Input( shape=(self.n_input, ) )
        
        dense  = Dense(
                        self.latentsize, input_shape=(self.n_input, ), 
                        activation=self.internal_activation, 
                        kernel_constraint = WeightClip(self.weight_clipping)
                )(inputs)
        
        if self.batch_norm_bool:
            dense = BatchNormalization()(dense)
            
        for l in range(self.layers-1):
            dense  = Dense(
                            self.latentsize, input_shape=(self.latentsize, ), 
                            activation=self.internal_activation, 
                            kernel_constraint = WeightClip(self.weight_clipping)
                    )(dense)
            
            if self.more_batch_norm_bool:
                dense = BatchNormalization()(dense)
#                 if l == 1:
#                     dense = BatchNormalization()(dense)
        
        
        if self.custom_activation_bool:
            output = Dense(
                        1, input_shape=(self.latentsize,), 
                        activation=custom_activation, 
                        kernel_constraint = WeightClip(self.weight_clipping)
                )(dense) 
        elif not self.custom_activation_bool:
            output = Dense(
                        1, input_shape=(self.latentsize,), 
                        activation='linear', 
                        kernel_constraint = WeightClip(self.weight_clipping)
                )(dense)
            
        
        model = Model(inputs=[inputs], outputs=[output])
        
        return model
        