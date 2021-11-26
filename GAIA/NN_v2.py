import tensorflow as tf
import h5py
import os
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow import Variable
from tensorflow import linalg as la
import numpy as np
import logging

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range                                                                                                
    '''
    def __init__(self, c=2):
        self.c = c
    def __call__(self, p):
        return tf.clip_by_value(p, clip_value_min=-self.c, clip_value_max=self.c)
    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}

class BSMfinder(Model):
    def __init__(self,input_shape, architecture=[1, 4, 1], weight_clipping=None, activation='sigmoid', name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        if weight_clipping:
            self.hidden_layers = [Dense(architecture[i+1], input_shape=(architecture[i],), activation=activation,
                                    kernel_constraint = WeightClip(weight_clipping)) for i in range(len(architecture)-2)]
            self.output_layer  = Dense(architecture[-1], input_shape=(architecture[-2],), activation='linear',
                                     kernel_constraint = WeightClip(weight_clipping))
        else:
            self.hidden_layers = [Dense(architecture[i+1], input_shape=(architecture[i],), activation=activation) for i in range(len(architecture)-2)]
            self.output_layer  = Dense(architecture[-1], input_shape=(architecture[-2],), activation='linear')
        self.build(input_shape)
        
    def call(self, x):
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x

class NPLMupgrade(Model):
    def __init__(self, input_shape, NUmatrix, NURmatrix, NU0matrix, SIGMAmatrix, N_Bkg, 
                 edgebinlist=[], means=[], points=[], A0matrix =[], A1matrix=[], A2matrix=[], binned_features=[], 
                 architecture=[1, 10, 1], weight_clipping=1., correction='BIN', ParNet_weights=None, train_nu=True, train_f=True, name=None, newpar=False,**kwargs):
        super().__init__(name=name, **kwargs)
        if (len(edgebinlist)!= len(binned_features) or  len(edgebinlist)!= len(means)  or len(means)!= len(binned_features)) and correction!='AN':
            logging.error('length of binned_features, means and edgebinlist must be the same')
            exit()
        if not correction in ['LI', 'BIN', 'PAR', 'AN']:
            logging.error("value %s for binning is not valid. Choose between '1D', '2D' and 'PAR' "%(binning))
            exit()
        if correction=='BIN' and len(edgebinlist)!=1:
            logging.error('BIN requires one element in edgebinlist and coefficient matrices')
            exit()
        if correction=='LI' and len(edgebinlist)!=1:
            logging.error('LI requires one element in edgebinlist and points')
            exit()
        if correction=='PAR' and ParNet_weights==None:
            logging.error("missing argument ParNet_weights required with 'PAR' correction")
            exit()
        if correction=='LI':
            self.expectation_layers = [LinearInterpolationExpLayer(input_shape, points[i], binning=binning) for i in range(len(points))]
            self.oi_layers   = [BinStepLayer(input_shape, edgebinlist[i], mean=means[i]) for i in range(len(edgebinlist))]
        elif correction == 'BIN':
            self.expectation_layers = [QuadraticExpLayer(input_shape, A0matrix[i], A1matrix[i], A2matrix[i]) for i in range(len(edgebinlist))]
            self.oi_layers   = [BinStepLayer(input_shape, edgebinlist[i], mean=means[i]) for i in range(len(edgebinlist))]
        elif correction == 'AN':
            self.expectation_layers = [AnalyticExpLayer(input_shape, means[i], newpar=newpar) for i in range(len(means))]
        self.binned_features = binned_features
        self.PositiveDef = False
        if correction =='PAR':
            configurationPar = ParNet_weights.split('/')[-3]
            configurationPar = configurationPar.split('_')[0].lower()
            self.configurationPar = configurationPar
            if 'PositiveDefinite' in ParNet_weights:
                self.PositiveDef = True
            architecturePar = ParNet_weights.split('layers', 1)[1]
            architecturePar = architecturePar.split('_act', 1)[0]
            architecturePar = architecturePar.split('_')
            layersPar       = [1]
            for layer in architecturePar:
                layersPar.append(int(layer))
            layersPar.append(1)
            inputsizePar    = layersPar[0]
            input_shapePar  = (None, inputsizePar)
            Delta_sb        = ParNet_weights.split('sigmaS', 1)[1]
            Delta_sb        = Delta_sb.split('_', 1)[0]
            input_examples  = ParNet_weights.split('sigmaS', 1)[1]
            input_examples  = input_examples.split('_patience')[0]
            input_examples  = input_examples.split('_')[1:]
            input_examples  = [float(s) for s in input_examples]
            self.Delta_std  = np.std(input_examples)
            activationPar   = ParNet_weights.split('act', 1)[1]
            activationPar   = activationPar.split('/', 1)[0]
            #wcPar           = ParNet_weights.split('wclip', 1)[1]
            #wcPar           = float(wcPar.split('/', 1)[0])
            self.Delta_sb   = float(Delta_sb) 
            self.Delta      = ParametricNet(input_shapePar, architecture=architecturePar, weight_clipping=None,activation=activationPar, configuration=configurationPar)
            self.Delta.load_weights(ParNet_weights)
            #don't want to train Delta
            for module in self.Delta.layers:
                for layer in module.layers:
                    layer.trainable = False

        self.nu   = Variable(initial_value=NUmatrix,         dtype="float32", trainable=train_nu,  name='nu')
        self.nuR  = Variable(initial_value=NURmatrix,        dtype="float32", trainable=False,     name='nuR')
        self.nu0  = Variable(initial_value=NU0matrix,        dtype="float32", trainable=False,     name='nu0')
        self.sig  = Variable(initial_value=SIGMAmatrix,      dtype="float32", trainable=False,     name='sigma')
        if train_f:
            self.f    = BSMfinder(input_shape, architecture, weight_clipping)
        self.N_Bkg   = N_Bkg
        self.train_f = train_f 
        self.correction = correction
        self.newpar = newpar
        self.build(input_shape)

    def call(self, x):
        nu      = tf.squeeze(self.nu)
        nuR     = tf.squeeze(self.nuR)
        nu0     = tf.squeeze(self.nu0)
        sigma   = tf.squeeze(self.sig)
        Laux    = tf.reduce_sum(-0.5*((nu-nu0)**2 - (nuR-nu0)**2)/sigma**2 )
        Laux    = Laux*tf.ones_like(x[:, 0:1])

        Lratio  = 0
        if self.correction in ['LI', 'BIN']:
            oi = []
            for i in range(len(self.binned_features)):
                oi.append(self.oi_layers[0].call(x[:, self.binned_features[i]:self.binned_features[i]+1]))
            ei      = self.N_Bkg * self.expectation_layers[0].call(self.nu[0:1, 0:1]-1)*(nu[1])
            eiR     = self.N_Bkg * self.expectation_layers[0].call(self.nuR[0:1, 0:1]-1)*(nuR[1])
            #if self.correction=='1D':
            Lratio = tf.matmul(oi[0], tf.math.log(ei/eiR))
            #elif self.correction=='2D':
                #Lratio = tf.reduce_sum(tf.multiply(tf.matmul(oi[1], tf.math.log(ei/eiR)), oi[0]), axis=1, keepdims=True)
        if self.correction == 'PAR':
            delta   = self.Delta.call(x)
            nus_std = (nu[0]-1)/self.Delta_sb
            nun_std = nu[1]
            nunR_std = nuR[1]
            if self.newpar:
                nus_std = nu[0]/(self.Delta_sb*self.Delta_std)#(tf.exp(nu[0])-1)/self.Delta_sb
                nun_std = tf.exp(nu[1])
                nunR_std = tf.exp(nuR[1])
            if self.PositiveDef:
                Lratio  = tf.math.log((1+delta[:, 0:1]*nus_std)**2 + (delta[:, 1:2]*nus_std)**2)
            else:
                for i in range(delta.shape[1]):
                    Lratio += delta[:, i:i+1]*nus_std**(i+1) #+ delta[:, 1:2]*nus_std**2
            Lratio += tf.math.log((tf.zeros_like(delta[:, 1:2])+nun_std)/(tf.zeros_like(delta[:, 1:2])+nunR_std))
        if self.correction == 'AN':
            nus = nu[0]*tf.ones_like(x)
            nusR = nuR[0]*tf.ones_like(x)
            nun = nu[1]*tf.ones_like(x)
            nunR = nuR[1]*tf.ones_like(x)
            '''
            if self.newpar:
                nus  = tf.exp(nu[0])*tf.ones_like(x)
                nusR = tf.exp(nuR[0])*tf.ones_like(x)
                nun  = tf.exp(nu[1])*tf.ones_like(x)
                nunR = tf.exp(nuR[1])*tf.ones_like(x)
            '''
            Lratio = self.expectation_layers[0].call([x, nus, nun, nusR, nunR])
        BSM     = tf.zeros_like(Laux)
        if self.train_f:
            BSM = self.f(x)
        output  = tf.keras.layers.Concatenate(axis=1)([BSM+Lratio, Laux])
        #self.add_metric(tf.reduce_mean(Laux), aggregation='mean', name='Laux')
        self.add_metric(tf.reduce_mean(nu[0]), aggregation='mean', name='scale')
        self.add_metric(tf.reduce_mean(nu[1]), aggregation='mean', name='norm')
        return output 

def NPLLoss(true, pred):
    f   = pred[:, 0]
    y   = true[:, 0]
    w   = true[:, 1]
    return tf.reduce_sum((1-y)*w*(tf.exp(f)-1) - y*w*(f))

def NPLLoss_New(true, pred):
    f   = pred[:, 0]
    Laux= pred[:, 1]
    y   = true[:, 0] 
    w   = true[:, 1] 
    return tf.reduce_sum((1-y)*w*(tf.exp(f)-1) - y*w*(f)) - tf.reduce_mean(Laux)

def CorrectionParLoss(true, pred):
    Lratio  = pred[:, 0]
    Laux    = pred[:, 1]
    y       = true[:, 0] 
    w       = true[:, 1]
    return tf.reduce_sum((1-y)*w*(tf.exp(Lratio)-1) - y*w*(Lratio)) - tf.reduce_mean(Laux)

def CorrectionBinLoss(true, pred):
    Lbinned = pred[:, 0]
    Laux    = pred[:, 1]
    N_R     = pred[:, 2]
    N       = pred[:, 3]
    y       = true[:, 0] # shape (batchsize,       )                                                                 
    w       = true[:, 1] # shape (batchsize,       )                                                                              
    return tf.reduce_sum(- y*w*(Lbinned)) - tf.reduce_mean(Laux) + tf.reduce_mean(N-N_R)

def LikelihoodLoss(true, pred):
    Lbinned = pred[:, 0]
    Laux    = pred[:, 1]
    N       = pred[:, 2]
    w       = true[:, 0]
    return - tf.reduce_sum(w*Lbinned) + tf.reduce_mean(N) - tf.reduce_mean(Laux)

def ParametricQuadraticLoss(true, pred):
    a1 = pred[:, 0]
    a2 = pred[:, 1]
    y  = true[:, 0]
    w  = true[:, 1]
    nu = true[:, 2]
    f  = tf.multiply(a1, nu) + tf.multiply(a2, nu**2)
    return tf.reduce_sum((1-y)*w*(tf.exp(f)-1) - y*w*(f))

def ParametricQuadraticPositiveDefiniteLoss(true, pred):
    a1 = pred[:, 0]
    a2 = pred[:, 1]
    y  = true[:, 0]
    w  = true[:, 1]
    nu = true[:, 2]
    f  = tf.math.log((tf.ones_like(a1)+tf.multiply(a1, nu))**2 + tf.multiply(a2**2, nu**2))
    #f  = (tf.multiply(a1, nu))**2 + tf.multiply(a2, nu**2)
    return tf.reduce_sum((1-y)*w*(tf.exp(f)-1) - y*w*(f))

def ParametricQuadraticPositiveDefiniteLoss_v2(true, pred):
    a1 = pred[:, 0]
    a2 = pred[:, 1]
    y  = true[:, 0]
    w  = true[:, 1]
    nu = true[:, 2]
    f  = (tf.ones_like(a1) + tf.multiply(a1, nu))**2 + tf.multiply(a2**2, nu**2) - tf.ones_like(a1)
    return tf.reduce_sum((1-y)*w*(tf.exp(f)-1) - y*w*(f))

def ReadFit_PTbins_from_h5(file_name='/eos/user/g/ggrosso/PhD/BSM/Sistematiche/MuMu/SM/PTbinsFits.h5'):
    f = h5py.File(file_name, 'r')
    q_b  = np.array(f.get('q_b'))
    q_e  = np.array(f.get('q_e'))
    m_sb = np.array(f.get('m_sb'))
    m_se = np.array(f.get('m_se'))
    m_eb = np.array(f.get('m_eb'))
    m_ee = np.array(f.get('m_ee'))
    c_sb = np.array(f.get('c_sb'))
    c_se = np.array(f.get('c_se'))
    c_eb = np.array(f.get('c_eb'))
    c_ee = np.array(f.get('c_ee'))
    bins = np.array(f.get('bins'))
    ee   = np.array(f.get('efficiency_endcaps'))
    eb   = np.array(f.get('efficiency_barrel'))
    se   = np.array(f.get('pt_scale_endcaps'))
    sb   = np.array(f.get('pt_scale_barrel'))
    f.close()
    return bins, sb, se, eb, ee, q_b, q_e, m_sb, m_se, m_eb, m_ee, c_sb, c_se, c_eb, c_ee

class BinStepLayer(Layer):
    def __init__(self, input_shape, edgebinlist, mean):
        super(BinStepLayer, self).__init__()
        self.edgebinlist = edgebinlist
        self.nbins       = edgebinlist.shape[0]-1
        self.w1          = np.zeros((2*self.nbins, 1))
        self.w2          = np.zeros((self.nbins, 2*self.nbins))
        self.b1          = np.zeros((2*self.nbins, 1))
        self.weight      = 100.

        # fix the weights and biases                                                                                                                                     
        for i in range(self.nbins+1):
            if i < self.nbins:
                for j in range(self.nbins*2):
                        self.w2[i, j]   =  0.
            if i==0:
                self.w1[2*i, 0] = self.weight
                self.b1[2*i]    = -1.*self.weight*self.edgebinlist[i]
                self.w2[i, 2*i] =  1.
            elif i==self.nbins:
                self.w1[2*i-1, 0] = self.weight
                self.b1[2*i-1]      = -1.*self.weight*self.edgebinlist[i]
                self.w2[i-1, 2*i-1] = -1.
            else:
                self.w1[2*i-1, 0] = self.weight
                self.b1[2*i-1]    = -1.*self.weight*self.edgebinlist[i]
                self.w1[2*i, 0]   = self.weight
                self.b1[2*i]      = -1.*self.weight*self.edgebinlist[i]
                self.w2[i, 2*i-1] =  1.
                self.w2[i-1, 2*i] = -1.

        self.w1 = Variable(initial_value=self.w1.transpose(), dtype="float32", trainable=False, name='w1' )
        self.w2 = Variable(initial_value=self.w2.transpose(), dtype="float32", trainable=False, name='w2' )
        self.b1 = Variable(initial_value=self.b1.transpose(), dtype="float32", trainable=False, name='b1' )
        self.mean = Variable(initial_value=mean, dtype="float32", trainable=False, name='mean' )
        self.build(input_shape)
        
    def call(self, x):
        x = tf.matmul(x*self.mean, self.w1) + self.b1
        x = keras.activations.relu(keras.backend.sign(x))
        x = tf.matmul(x, self.w2)
        return x

class AnalyticExpLayer(Layer):
    def __init__(self, input_shape, mean, newpar=False):
        super(AnalyticExpLayer, self).__init__()
        self.build(input_shape)
        self.mean = mean
        self.newpar = newpar

    def call(self, x):
        nuS = x[1]
        nuN = x[2]
        nuSR= x[3]
        nuNR= x[4]
        x   = x[0]*self.mean
        if self.newpar:
            return -8*tf.multiply(x,(tf.exp(-1*nuS)-tf.exp(-1*nuSR)))+nuSR+nuN-nuS-nuNR
        else:
            return -8*tf.multiply(x,(1./nuS-1./nuSR))+tf.math.log(nuSR)+tf.math.log(nuN)-tf.math.log(nuS)-tf.math.log(nuNR)

class LinearExpLayer(Layer):
    def __init__(self, input_shape, A0matrix, A1matrix, endcaps_barrel_r):
        super(LinearExpLayer, self).__init__()
        self.a0         = Variable(initial_value=A0matrix[0, :],   dtype="float32", trainable=False, name='a0' )
        self.a1_barrel  = Variable(initial_value=A1matrix[:2, :],  dtype="float32", trainable=False, name='a1b' )
        self.a1_endcaps = Variable(initial_value=A1matrix[2:, :],  dtype="float32", trainable=False, name='a1e' )
        self.ebr        = Variable(initial_value=endcaps_barrel_r, dtype="float32", trainable=False, name='eb_ratio' )
        self.build(input_shape)

    def call(self, x):
        x = tf.matmul(x, self.a1_barrel) +  tf.matmul(tf.math.multiply(self.ebr, x), self.a1_endcaps) + self.a0
        return x # e_j(nu_1,..., nu_i) # [B x Nbins]
'''
class QuadraticExpLayer(Layer):
    def __init__(self, input_shape, A0matrix, A1matrix, A2matrix, endcaps_barrel_r):
        super(QuadraticExpLayer, self).__init__()
        self.a0         = Variable(initial_value=A0matrix[0, :],   dtype="float32", trainable=False, name='a0' )
        self.a1_barrel  = Variable(initial_value=A1matrix[:2, :],  dtype="float32", trainable=False, name='a1b' )
        self.a2_barrel  = Variable(initial_value=A2matrix[:2, :],  dtype="float32", trainable=False, name='a2b' )
        self.a1_endcaps = Variable(initial_value=A1matrix[2:, :],  dtype="float32", trainable=False, name='a1e' )
        self.a2_endcaps = Variable(initial_value=A2matrix[2:, :],  dtype="float32", trainable=False, name='a2e' )
        self.ebr        = Variable(initial_value=endcaps_barrel_r, dtype="float32", trainable=False, name='eb_ratio' )
        self.build(input_shape)

    def call(self, x):
        x_e = tf.math.multiply(self.ebr, x)
        y   = (self.a0 + tf.matmul(x, self.a1_barrel) + tf.matmul(x_e, self.a1_endcaps)) 
        y   = tf.math.multiply(y, 1+ tf.matmul(tf.math.multiply(x, x), tf.math.divide_no_nan(self.a2_barrel, self.a0)))
        y   = tf.math.multiply(y, 1+ tf.matmul(tf.math.multiply(x_e, x_e), tf.math.divide_no_nan(self.a2_endcaps, self.a0)))
        return y # e_j(nu_1,..., nu_i) # [B x Nbins]
'''
class QuadraticExpLayer(Layer):
    def __init__(self, input_shape, A0matrix, A1matrix, A2matrix):
        super(QuadraticExpLayer, self).__init__()
        self.a0  = Variable(initial_value=np.expand_dims(A0matrix, axis=1),  dtype="float32", trainable=False, name='a0' )
        self.a1  = Variable(initial_value=np.expand_dims(A1matrix, axis=1),  dtype="float32", trainable=False, name='a1' )
        self.a2  = Variable(initial_value=np.expand_dims(A2matrix, axis=1),  dtype="float32", trainable=False, name='a2' )
        self.build(input_shape)

    def call(self, x):
        
        y   = self.a0 + tf.matmul(self.a1, x) + tf.matmul(self.a2, tf.matmul(x, x))
        return y # e_j(nu_1,..., nu_i) # [B x Nbins]                                                                                                        

class LinearInterpolationExpLayer(Layer):
    def __init__(self, input_shape, points):
        super(LinearInterpolationExpLayer, self).__init__()
        self.nbins       = points.shape[0]
        self.npoints     = points.shape[1]
        self.x           = points[0, :, 0]
        self.a1          = (points[:,1:, 1]-points[:,:-1, 1])/(points[:,1:, 0]-points[:,:-1, 0])
        self.a0          = (points[:,:-1, 1] - self.a1*points[:,:-1, 0]) +1e-10
    
        self.b1          = np.zeros((self.npoints, 1))
        self.w1          = np.zeros((self.npoints, 1))
        self.w2          = np.zeros((self.npoints-1, self.npoints))
        self.b2          = np.zeros((self.npoints-1, 1))
        self.weight      = 100.
        
        for i in range(self.npoints):
            if i==0 or i==(self.npoints-1):
                continue
            self.w1[i, 0] = self.weight
            self.b1[i, 0] = -1*self.weight*self.x[i]
        for i in range(self.npoints-1):
            self.w2[i, i]   = 1.
            self.w2[i, i+1] = -1.
            if i==0:
                self.b2[i, 0] = 1

        self.w1 = Variable(initial_value=self.w1, dtype="float32", trainable=False, name='w1' )
        self.w2 = Variable(initial_value=self.w2, dtype="float32", trainable=False, name='w2' )
        self.b1 = Variable(initial_value=self.b1, dtype="float32", trainable=False, name='b1' )
        self.b2 = Variable(initial_value=self.b2, dtype="float32", trainable=False, name='b2' )
        self.a0 = Variable(initial_value=self.a0, dtype="float32", trainable=False, name='a0' )
        self.a1 = Variable(initial_value=self.a1, dtype="float32", trainable=False, name='a1' )
        self.build(input_shape)
            
    def call(self, nu):
        scale = nu[:, 0:1] # [1, 1]
        norm  = nu[:, 1:2] # [1, 1]
        y = tf.matmul(self.w1, scale)+self.b1
        y = keras.activations.relu(keras.backend.sign(y)) # [npoints, 1]
        proj       = tf.matmul(self.w2, y) + self.b2  # [npoints-1, 1]
        proj_scale = tf.matmul(proj, scale) # [npoints-1, 1]
        return tf.multiply(tf.matmul(self.a1, proj_scale)+tf.matmul(self.a0, proj), tf.ones((self.nbins, 1))+tf.matmul(tf.ones((self.nbins, 1)), norm))
'''
class LinearInterpolationExpLayer(Layer):
    def __init__(self, input_shape, points, binning='1D'):
        super(LinearInterpolationExpLayer, self).__init__()
        self.nbins       = points.shape[0]
        print('nbins = %i'%(self.nbins))
        self.npoints     = points.shape[1]
        self.matrix_size = self.nbins
        self.binning     = binning
        if binning=='2D':
            self.matrix_size = int((np.sqrt(1+8*self.nbins)-1)*0.5)
            print('n = %i'%(self.matrix_size))
            self.sequence1 = [0]+[(1+n)*self.matrix_size-np.sum(np.arange(n+1)) for n in np.arange(self.matrix_size)]
            self.sequence2 = [1+np.sum(np.arange(n+1)) for n in range(self.matrix_size)]
            #print(self.sequence1)
            #print(self.sequence2)
            self.upper_triangle_proj = la.band_part(tf.ones((self.matrix_size, self.matrix_size)), 0, -1)
            self.upper_triangle_proj+= 1e-10*(tf.ones((self.matrix_size, self.matrix_size))-self.upper_triangle_proj)
            self.indices   = np.array([])
            for i in range(self.matrix_size-1):
                self.indices = np.append(self.indices, np.arange(self.sequence1[i], self.sequence1[i+1]))
                self.indices = np.append(self.indices, -1*np.arange(self.sequence2[i], self.sequence2[i+1]))
                if self.indices.shape[0]>=self.nbins:
                    self.indices = self.indices[:self.nbins].astype(int)
                    break

        self.iu          = np.triu_indices(self.matrix_size)
        self.x           = points[0, :, 0]
        self.a1          = (points[:,1:, 1]-points[:,:-1, 1])/(points[:,1:, 0]-points[:,:-1, 0])
        self.a0          = (points[:,:-1, 1] - self.a1*points[:,:-1, 0]) +1e-10
        self.a1          = self.a1[self.indices, :]
        self.a0          = self.a0[self.indices, :]
        self.b1          = np.zeros((self.npoints, 1))
        self.w1          = np.zeros((self.npoints, 1))
        self.w2          = np.zeros((self.npoints-1, self.npoints))
        self.b2          = np.zeros((self.npoints-1, 1))
        self.weight      = 100.

        for i in range(self.npoints):
            if i==0 or i==(self.npoints-1):
                continue
            self.w1[i, 0] = self.weight
            self.b1[i, 0] = -1*self.weight*self.x[i]
        for i in range(self.npoints-1):
            self.w2[i, i]   = 1.
            self.w2[i, i+1] = -1.
            if i==0:
                self.b2[i, 0] = 1

        self.w1 = Variable(initial_value=self.w1, dtype="float32", trainable=False, name='w1' )
        self.w2 = Variable(initial_value=self.w2, dtype="float32", trainable=False, name='w2' )
        self.b1 = Variable(initial_value=self.b1, dtype="float32", trainable=False, name='b1' )
        self.b2 = Variable(initial_value=self.b2, dtype="float32", trainable=False, name='b2' )
        self.a0 = Variable(initial_value=self.a0, dtype="float32", trainable=False, name='a0' )
        self.a1 = Variable(initial_value=self.a1, dtype="float32", trainable=False, name='a1' )
        self.build(input_shape)

    def call(self, nu):
        scale = nu[:, 0:1] # [1, 1]                                                                                                                             
        norm  = nu[:, 1:2] # [1, 1]                                                                                                                             
        y = tf.matmul(self.w1, scale)+self.b1
        y = keras.activations.relu(keras.backend.sign(y)) # [npoints, 1]                                                                                        
        proj       = tf.matmul(self.w2, y) + self.b2  # [npoints-1, 1]                                                                                         
        proj_scale = tf.matmul(proj, scale) # [npoints-1, 1]                                                                                                   
        e  = tf.multiply(tf.matmul(self.a1, proj_scale)+tf.matmul(self.a0, proj), tf.ones((self.nbins, 1))+tf.matmul(tf.ones((self.nbins, 1)), norm))

        if self.binning=='1D':
            return e
        else:
            e = tf.squeeze(e)
            output = tf.concat([e, e[self.matrix_size:][::-1]], 0)
            output = tf.reshape(output, (self.matrix_size, self.matrix_size))
            return tf.multiply(self.upper_triangle_proj, output)

'''
class ParametricNet(Model):
    def __init__(self, input_shape, architecture=[1, 10, 1], weight_clipping=None, activation='sigmoid', configuration='linear', name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.a1 = BSMfinder(input_shape, architecture, weight_clipping, activation=activation)
        if configuration=='quadratic':
            self.a2 = BSMfinder(input_shape, architecture, weight_clipping, activation=activation)
        
        self.configuration = configuration
        self.build(input_shape)

    def call(self, x):
        a1 = self.a1(x)
        a2 = tf.zeros_like(a1)
        if self.configuration=='quadratic':
            a2 = self.a2(x)
        output  = tf.keras.layers.Concatenate(axis=1)([a1, a2])
        return output

class DeltaParametric(Model):
    def __init__(self, input_shape, NUmatrix, NURmatrix, NU0matrix, SIGMAmatrix, ParWeightsH5, architecture = [2, 5, 1], weight_clipping=1, activation='sigmoid', configuration='quadratic', name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = ParametricNet(input_shape, architecture, weight_clipping, activation, configuration)
        self.nu    = Variable(initial_value=NUmatrix,    dtype="float32", trainable=True,  name='nu')
        self.nuR   = Variable(initial_value=NURmatrix,   dtype="float32", trainable=False, name='nuR')
        self.nu0   = Variable(initial_value=NU0matrix,   dtype="float32", trainable=False, name='nu0')
        self.sig   = Variable(initial_value=SIGMAmatrix, dtype="float32", trainable=False, name='sigma')
        self.delta.load_weights(ParWeightsH5)
        # don't want to train delta
        for module in self.delta.layers:
            for layer in module.layers:
                layer.trainable = False
        self.build(input_shape)
    
    def call(self, x):
        nu      = tf.squeeze(self.nu)
        nuR     = tf.squeeze(self.nuR)
        nu0     = tf.squeeze(self.nu0)
        sigma   = tf.squeeze(self.sig)
        delta   = self.delta.call(x[:, 0:2])
        Lratio  = delta[:, 0:1]*nu[0]/sigma[0]  + delta[:, 1:2]*nu[0]/sigma[0] *nu[0]/sigma[0] 
        Lratio += tf.math.log((tf.ones_like(delta[:, 1:2])+nu[1])/(tf.ones_like(delta[:, 1:2])+nuR[1]))
        Laux    = tf.reduce_sum(-0.5*((nu-nu0)**2 - (nuR-nu0)**2)/sigma**2 )# Gaussian (scalar value)
        Laux    = Laux*tf.ones_like(Lratio)
        output  = tf.keras.layers.Concatenate(axis=1)([Lratio, Laux])
        self.add_metric(tf.reduce_mean(Laux), aggregation='mean', name='Laux')
        self.add_metric(nu[0], aggregation='mean', name='scale_barrel')
        self.add_metric(nu[1], aggregation='mean', name='efficiency_barrel')
        return output

class DeltaQuadratic(Model):
    def __init__(self, input_shape, N_D, N_R, edgebinlist, means, A0matrix, A1matrix, A2matrix, endcaps_barrel_r, NUmatrix, NURmatrix, NU0matrix, SIGMAmatrix, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.oi  = BinStepLayer(input_shape, edgebinlist, means)
        self.ei  = QuadraticExpLayer(input_shape, A0matrix, A1matrix, A2matrix, endcaps_barrel_r)
        self.eiR = QuadraticExpLayer(input_shape, A0matrix, A1matrix, A2matrix, endcaps_barrel_r)
        self.nu  = Variable(initial_value=NUmatrix,         dtype="float32", trainable=True,  name='nu')
        self.nuR = Variable(initial_value=NURmatrix,        dtype="float32", trainable=False, name='nuR')
        self.nu0 = Variable(initial_value=NU0matrix,        dtype="float32", trainable=False, name='nu0')
        self.sig = Variable(initial_value=SIGMAmatrix,      dtype="float32", trainable=False, name='sigma')
        self.N_D = N_D
        self.N_R = N_R
        self.build(input_shape)
    
    def call(self, x):
        oi      = self.oi.call(x) * 0.5
        nu      = tf.squeeze(self.nu)
        nuR     = tf.squeeze(self.nuR)
        nu0     = tf.squeeze(self.nu0)
        sigma   = tf.squeeze(self.sig)
        ei      = tf.transpose(self.ei.call(tf.transpose(self.nu)))
        eiR     = tf.transpose(self.eiR.call(tf.transpose(self.nuR)))
        Lbinned = tf.matmul(oi, tf.math.log(ei/eiR))# (batchsize, 1)
        N_R     = tf.reduce_sum(eiR) *self.N_D
        N       = tf.reduce_sum(ei)  *self.N_D
        Laux    = tf.reduce_sum(-0.5*((nu-nu0)**2 - (nuR-nu0)**2)/sigma**2 )# Gaussian (scalar value)
        Laux    = Laux*tf.ones_like(Lbinned)
        N_R     = N_R*tf.ones_like(Lbinned)
        N       = N*tf.ones_like(Lbinned)
        output  = tf.keras.layers.Concatenate(axis=1)([Lbinned, Laux, N_R, N])
        self.add_metric(tf.reduce_mean(Laux), aggregation='mean', name='Laux')
        self.add_metric(tf.reduce_mean(N),    aggregation='mean', name='N')
        self.add_metric(tf.reduce_mean(N_R),  aggregation='mean', name='N_R')
        self.add_metric(nu[0], aggregation='mean', name='scale_barrel')
        self.add_metric(nu[1], aggregation='mean', name='efficiency_barrel')
        return output

class DeltaLinearInterpolation(Model):
    def __init__(self, input_shape, N_D, N_R, means, edgebinlist, points,  NUmatrix, NURmatrix, NU0matrix, SIGMAmatrix, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.oi  = BinStepLayer(input_shape, edgebinlist, means)
        self.ei  = LinearInterpolationExpLayer(input_shape, points)
        self.eiR = LinearInterpolationExpLayer(input_shape, points)
        self.nu  = Variable(initial_value=NUmatrix,         dtype="float32", trainable=True,  name='nu')
        self.nuR = Variable(initial_value=NURmatrix,        dtype="float32", trainable=False, name='nuR')
        self.nu0 = Variable(initial_value=NU0matrix,        dtype="float32", trainable=False, name='nu0')
        self.sig = Variable(initial_value=SIGMAmatrix,      dtype="float32", trainable=False, name='sigma')
        self.N_D = N_D
        self.N_R = N_R
        self.build(input_shape)

    def call(self, x):
        oi      = self.oi.call(x) * 0.5
        nu      = tf.squeeze(self.nu)
        nuR     = tf.squeeze(self.nuR)
        nu0     = tf.squeeze(self.nu0)
        sigma   = tf.squeeze(self.sig)
        ei      = self.ei.call(tf.transpose(self.nu))
        eiR     = self.eiR.call(tf.transpose(self.nuR))
        Lbinned = tf.matmul(oi, tf.math.log(ei/eiR))
        N_R     = tf.reduce_sum(eiR) *self.N_D
        N       = tf.reduce_sum(ei)  *self.N_D
        Laux    = tf.reduce_sum(-0.5*((nu-nu0)**2 - (nuR-nu0)**2)/sigma**2 )
        Laux    = Laux*tf.ones_like(Lbinned)
        N_R     = N_R*tf.ones_like(Lbinned)
        N       = N*tf.ones_like(Lbinned)
        output  = tf.keras.layers.Concatenate(axis=1)([Lbinned, Laux, N_R, N])
        self.add_metric(tf.reduce_mean(Laux), aggregation='mean', name='Laux')
        self.add_metric(tf.reduce_mean(N),    aggregation='mean', name='N')
        self.add_metric(tf.reduce_mean(N_R),  aggregation='mean', name='N_R')
        self.add_metric(nu[0], aggregation='mean', name='scale_barrel')
        self.add_metric(nu[1], aggregation='mean', name='efficiency_barrel')
        return output


class LikelihoodLinearInterpolation(Model):
    def __init__(self, input_shape, N_B, N_S, edgebinlist, points_BKG, points_SIG, MUmatrix, NUmatrix, NU0matrix, SIGMAmatrix, train_mu=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.oi  = BinStepLayer(input_shape, edgebinlist=edgebinlist, mean=[1.])
        self.eiB = LinearInterpolationExpLayer(input_shape, points_BKG)
        self.eiS = LinearInterpolationExpLayer(input_shape, points_SIG)
        self.nu  = Variable(initial_value=NUmatrix,         dtype="float32", trainable=True,  name='nu')
        self.mu  = Variable(initial_value=MUmatrix,         dtype="float32", trainable=train_mu, name='mu')
        self.nu0 = Variable(initial_value=NU0matrix,        dtype="float32", trainable=False, name='nu0')
        self.sig = Variable(initial_value=SIGMAmatrix,      dtype="float32", trainable=False, name='sigma')
        self.N_B = N_B
        self.N_S = N_S
        self.build(input_shape)

    def call(self, x):
        oi      = self.oi.call(x)
        nu      = tf.squeeze(self.nu)
        mu      = tf.squeeze(self.mu)
        nu0     = tf.squeeze(self.nu0)
        sigma   = tf.squeeze(self.sig)
        eiB     = self.eiB.call(self.nu)
        eiS     = self.eiS.call(self.nu)
        Lbinned = tf.matmul(oi, tf.math.log(mu[0]*self.N_B*eiB + mu[1]*self.N_S*eiS))
        Laux    = tf.reduce_sum(-0.5*(nu-nu0)**2/sigma**2)
        Laux    = Laux*tf.ones_like(Lbinned)
        N       = tf.reduce_sum(self.N_B*eiB + self.mu*self.N_S*eiS)
        N       = N*tf.ones_like(Lbinned)
        output  = tf.keras.layers.Concatenate(axis=1)([Lbinned, Laux, N])
        self.add_metric(tf.reduce_mean(Laux), aggregation='mean', name='Laux')
        self.add_metric(tf.reduce_mean(N),    aggregation='mean', name='N')
        self.add_metric(mu[1], aggregation='mean', name='strenght')
        self.add_metric(nu[0], aggregation='mean', name='scale_barrel')
        self.add_metric(nu[1], aggregation='mean', name='efficiency_barrel')
        return output


def collect_history(file_path, key='scale'):
    ''' 
    the function collects the history of the loss and saves t=-2*loss at the check points.
    
    files_id: array of toy labels 
    DIR_IN: directory where all the toys' outputs are saved
    patience: interval between two check points (epochs)
    
    The function returns a 2D-array with final shape (nr toys, nr check points).
    '''
    if not os.path.exists(file_path):
        print('file does not exist')
        return None
    f = h5py.File(file_path, 'r')
    check = f.get(key)
    if not check:
        print("key does not exist")
        return None
    check = np.array(check)
    f.close()
    return check

def Read_Points(file_path):
    f      = h5py.File(file_path, 'r')
    points = [np.array(f.get('points'))]
    bins   = []
    if 'bins' in f.keys():
        bins.append(np.array(f.get('bins')))
    if 'bins1' in f.keys():
        bins.append(np.array(f.get('bins1')))
    if 'bins2' in f.keys():
        bins.append(np.array(f.get('bins2')))
    ee     = np.array(f.get('efficiency_endcaps'))
    eb     = np.array(f.get('efficiency_barrel'))
    se     = np.array(f.get('pt_scale_endcaps'))
    sb     = np.array(f.get('pt_scale_barrel'))
    f.close()
    return points, bins, sb, se, eb, ee

def Read_FitBins(filename):
    f = h5py.File(filename, "r")
    q = np.array(f.get("q"))
    m = np.array(f.get("m"))
    c = np.array(f.get("c"))
    b = np.array(f.get("bins"))
    n = np.array(f.get("nuisance"))
    f.close()
    return q, m, c, b, n
