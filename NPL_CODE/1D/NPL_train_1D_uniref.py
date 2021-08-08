#python code to run 5D test: Zprime vs. Zmumu
#from __future__ import division                                                                                                                                                  
import numpy as np
import argparse
import os
import h5py
import time
import datetime
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import model_from_json
import tensorflow as tf
from scipy.stats import norm, expon, chi2, uniform
from scipy.stats import chisquare
from keras.constraints import Constraint, max_norm
from keras import callbacks
from keras import metrics, losses, optimizers
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, Conv1D, Flatten, Dropout, LeakyReLU, Layer                                                                                      
from keras.utils import plot_model

from Data_Reader import BuildSample_DY
from NPL_Model import NPL_Model

#ARGS:
parser = argparse.ArgumentParser()    #Python tool that makes it easy to create an user-friendly command-line interface
parser.add_argument('-o','--output', type=str, help="output directory", required=True)
parser.add_argument('-i','--input', type=str, help="input file", required=True)
parser.add_argument('-t','--toys', type=str, help="number of repetitions", required=True)
parser.add_argument('-sig','--signal', type=int, help="number of signal events", required=True)
parser.add_argument('-bkg','--background', type=int, help="number of background events", required=True)
parser.add_argument('-ref','--reference', type=int, help="number of reference events", required=True)
parser.add_argument('-epochs','--epochs', type=int, help="number of epochs", required=True)
parser.add_argument('-latsize','--latsize', type=int, help="number of nodes in each hidden layer", required=True)
parser.add_argument('-layers','--layers', type=int, help="number of layers", required=True)
parser.add_argument('-wclip','--weight_clipping', type=float, help="weight clipping", required=True)
args = parser.parse_args()
#output path
OUTPUT_PATH = args.output
#toy
toy = args.toys
#signal path
INPUT_PATH_SIG = args.input#'/lustre/cmswork/dalsanto/1D_toy/1D_experiments/H1'#args.input #sys.argv[3] #EX: '/eos/project/d/dshep/BSM_Detection/DiLepton_Zprime300'

#background path     
INPUT_PATH_BKG = '/lustre/cmswork/dalsanto/1D_toy/1D_experiments/H0/' #'/eos/project/d/dshep/BSM_Detection/DiLepton_SM/'

#random seed
seed=datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)
print('Random seed:'+str(seed))

N_Sig = args.signal
N_Bkg = args.background
N_ref = args.reference

N_Sig_P = np.random.poisson(lam=N_Sig, size=1)
N_Sig_p = N_Sig_P[0]
print('N_Sig: '+str(N_Sig))
print('N_Sig_Pois: '+str(N_Sig_p))

N_Bkg_P = np.random.poisson(lam=N_Bkg, size=1)
N_Bkg_p = N_Bkg_P[0]
print('N_Bkg: '+str(N_Bkg))
print('N_Bkg_Pois: '+str(N_Bkg_p))

total_epochs = args.epochs #00 #CAMBIAREEEEE
latentsize = args.latsize # number of nodes in each hidden layer
layers = args.layers #number of hidden layers

patience = 5000 # number of epochs between two consecutives saving points
nfile_REF = 1 #number of files in REF repository
nfile_SIG = 1 #number of files in SIG repository

#GLOBAL VARIABLES:
weight_clipping = args.weight_clipping
N_D = N_Bkg
N_R = N_ref

# define output path
ID = '/1D_patience'+str(patience)+'_ref'+str(N_ref)+'_bkg'+str(N_Bkg)+'_sig'+str(N_Sig)+'_epochs'+str(total_epochs)+'_latent'+str(latentsize)+'_layers'+str(layers)+'_wclip'+str(weight_clipping)
OUTPUT_PATH = OUTPUT_PATH+ID
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
OUTPUT_FILE_ID = '/Toy1D_patience'+str(patience)+'_'+str(N_ref)+'ref_'+str(N_Bkg)+'_'+str(N_Sig)+'_'+str(toy)

#check if the toy label has already been used. If it is so, exit the program without repeating.
if os.path.isfile(f"{OUTPUT_PATH}/{OUTPUT_FILE_ID}.txt"):
    exit()

print('Start analyzing Toy '+toy)

#---------------------------------------
start_time=time.time()

#Read Data

#BACKGROUND+REFERENCE
HLF_REF = np.expand_dims(np.linspace(0,1.8,N_ref),axis=1)
HLF_BKG = BuildSample_DY(N_Events=N_Bkg_p, INPUT_PATH=INPUT_PATH_BKG, seed=seed, nfiles=nfile_REF)

#SIGNAL                                                                  
INPUT_PATH_SIG = INPUT_PATH_SIG + "/"
print("1D input path: "+INPUT_PATH_SIG)
HLF_SIG = BuildSample_DY(N_Events=N_Sig_p, INPUT_PATH=INPUT_PATH_SIG, seed=seed, nfiles=nfile_SIG)
print(HLF_REF.shape)

#TARGETS
target_REF = np.zeros(N_ref)
target_DATA = np.ones(N_Bkg_p+N_Sig_p)
target = np.append(target_REF, target_DATA)
target = np.expand_dims(target, axis=1)

weights_REF = expon.pdf(HLF_REF,loc=0,scale=0.125)*1.8*N_Bkg/N_ref
weights_DATA = np.ones(N_Bkg_p+N_Sig_p)
weights = np.append(weights_REF,weights_DATA)
weights = np.expand_dims(weights, axis=1)

feature = np.concatenate((HLF_REF, HLF_BKG, HLF_SIG), axis=0)
feature = np.concatenate((feature, target, weights), axis=1)
np.random.shuffle(feature)
print(f'feature shape: {feature.shape}')
target = feature[:, -2]
weights = feature[:, -1]
feature = feature[:, :-2]

plt.figure(figsize=(7,7))
plt.hist(HLF_REF,bins=50,range=(0,1.5),weights=weights_REF,label='Ref')
plt.hist(np.concatenate((HLF_BKG, HLF_SIG), axis=0),bins=50,range=(0,1.5),weights=weights_DATA,label='Data')
plt.legend(loc='best')
plt.savefig(f"{OUTPUT_PATH}/{OUTPUT_FILE_ID}_histo.png")

#remove MASS from the input features
#feature = feature[:, :-1]

#standardize dataset                                                                                                                                                                                                  
for j in range(feature.shape[1]):
    vec = feature[:, j]
    mean = np.mean(vec)
    std = np.std(vec)
    if np.min(vec) < 0:
        #save the normalization parameters
        with open(OUTPUT_PATH+OUTPUT_FILE_ID+'_normalization.txt','w') as norm_file:
            norm_file.write(str(f'Standardization -> Mean: {mean}, Std: {std}'))
        vec = vec-mean
        vec = vec/std
    elif np.max(vec) > 1.0:# Assume data is exponential -- just set mean to 1.
        #save the normalization parameters
        with open(OUTPUT_PATH+OUTPUT_FILE_ID+'_normalization.txt','w') as norm_file:
            norm_file.write(str(f'Exponential standardization -> Mean: {mean}'))
        vec = vec *1./ mean
    feature[:, j] = vec
    


print(f'Target: {target.shape}')
print(f'Features: {feature.shape}')
print('Start training Toy '+toy)
#--------------------------------------------------------------------
def double_logger(stringa):
    print(stringa)
    with open(f"{OUTPUT_PATH}/check_seed{seed}.txt", "a") as norm_file:
        norm_file.write(str(stringa+'\n'))
        
#Loss function definition                                                                                                                                                                                             
def Loss(yTrue, yPred):
    return K.sum(-1.*yTrue*(yPred) + (1-yTrue)*N_D/float(N_R)*(K.exp(yPred)-1))

def LossWeighted(yTrue, yPred, w):
    return K.sum( (1-yTrue)*w*(K.exp(yPred)-1) - yTrue*w*yPred) #N_D/float(N_R)

'''
#Define the callback
class myCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % int(total_epochs/20) == 0:
            #print(f"Epoch: {epoch}, Logs:" ,logs)
            with open(f"{OUTPUT_PATH}/check.txt", "a") as myfile:
                myfile.write(f"Epoch: {epoch}, Logs: {logs}\n")
'''       
            
# training
def training_epoch(train_x,train_y,w):
    with tf.GradientTape() as tape:
        predicted_y = tf.squeeze(BSMfinder(train_x, training=True))
        loss_value = LossWeighted(train_y,predicted_y,weights)
    gradients = tape.gradient(loss_value, BSMfinder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, BSMfinder.trainable_variables))
    return loss_value.numpy()
    
# training
BSMfinder = NPL_Model(feature.shape[1], latentsize, layers, weight_clipping)
print(BSMfinder.summary())
'''
BSMfinder.compile(loss = LossWeighted,  optimizer = 'adam')
hist = BSMfinder.fit(feature, target, batch_size=batch_size, epochs=total_epochs,verbose=0,callbacks=[myCallback()],)
print('Finish training Toy '+toy)
'''
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
dataset = tf.data.Dataset.from_tensor_slices((feature.astype('float32'), target.astype('float32'), weights.astype('float32')))
batched_dataset = dataset.batch(batch_size=feature.shape[0])
loss_list=[]
for epoch_num in range(total_epochs):
    for (train_x,train_y,w) in batched_dataset:
        loss_value = training_epoch(train_x,train_y,w)
    if (np.isnan(loss_value)) | (np.isinf(loss_value)):
        double_logger(f'Exiting due to inf/nan in the loss')
        break
    loss_list.append(loss_value)
    if epoch_num%patience==0: double_logger(f'Epoch {epoch_num}, Loss {loss_value:.4f}')
print('Finish training Toy '+toy)

'''
Data = feature[target==1]
Reference = feature[target!=1]
print(Data.shape, Reference.shape)                                                                                                                            

# inference
f_Data = BSMfinder.predict(Data, batch_size=batch_size)
f_Reference = BSMfinder.predict(Reference, batch_size=batch_size)
f_All = BSMfinder.predict(feature, batch_size=batch_size)
'''
# metrics                                                                                                                                                           
loss = np.array(loss_list)

# test statistic                                                                                                                                              
final_loss=loss[-1]
t_e_OBS = -2*final_loss

# save t                                                                                                                                                            
log_t = OUTPUT_PATH+OUTPUT_FILE_ID+'_t.txt'
out = open(log_t,'w')
out.write("%f\n" %(t_e_OBS))
out.close()

# write the loss history                                                                                                     
log_history =OUTPUT_PATH+OUTPUT_FILE_ID+'_history'+str(patience)+'.h5'
f = h5py.File(log_history,"w")
keepEpoch = np.array(range(total_epochs))
keepEpoch = keepEpoch % patience == 0
f.create_dataset('loss', data=loss[keepEpoch], compression='gzip')
f.close()

# save the model                                                                                                                                                  
log_model = OUTPUT_PATH+OUTPUT_FILE_ID+'_seed'+str(seed)+'_model.json'
log_weights = OUTPUT_PATH+OUTPUT_FILE_ID+'_seed'+str(seed)+'_weights.h5'
model_json = BSMfinder.to_json()
with open(log_model, "w") as json_file:
    json_file.write(model_json)
BSMfinder.save_weights(log_weights)


'''
# save outputs                                                                                                                                                      
log_predictions =OUTPUT_PATH+OUTPUT_FILE_ID+'_predictions.h5'
f = h5py.File(log_predictions,"w")
f.create_dataset('feature', data=f_All, compression='gzip')
f.create_dataset('target', data=target, compression='gzip')
f.close()
'''

print('Output saved for Toy '+toy)
double_logger(f"Execution time: {time.time()-start_time}")
double_logger('----------------------------\n')