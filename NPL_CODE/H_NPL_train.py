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

from H_Data_Reader import BuildSample_DY_Higgs
from H_NPL_Model import NPL_Model

#ARGS:
parser = argparse.ArgumentParser()    #Python tool that makes it easy to create an user-friendly command-line interface
parser.add_argument('-o','--output', type=str, help="output directory", required=True) # 
#parser.add_argument('-i','--input', type=str, help="input file", required=True)
parser.add_argument('-t','--toys', type=str, help="number of repetitions", required=True)
parser.add_argument('-sig','--signal', type=str, help="if signal", required=True)
parser.add_argument('-DY','--DY', type=str, help='if DY+jets bkg', required=True)
parser.add_argument('-CMS','--CMS', type=str, help='if data or MC', required=True)
#parser.add_argument('-bkg','--background', type=int, help="number of background events", required=True)
parser.add_argument('-frac_ref','--frac_ref', type=float, help="Fraction of reference events in the dataset", required=True)
#parser.add_argument('-frac_bkg','--frac_bkg', type=float, help="Fraction of background events in the dataset", required=True)
#parser.add_argument('-frac_sig','--frac_sig', type=float, help="Fraction of background events in the dataset", required=True)
parser.add_argument('-epochs','--epochs', type=int, help="number of epochs", required=True)
parser.add_argument('-latsize','--latsize', type=int, help="number of nodes in each hidden layer", required=True)
parser.add_argument('-layers','--layers', type=int, help="number of layers", required=True)
parser.add_argument('-wclip','--weight_clipping', type=float, help="weight clipping", required=True)
parser.add_argument('-act','--internal_activation',type=str, help='internal activation', required=True)
args = parser.parse_args()

#toy
toy = args.toys 
internal_activation = args.internal_activation

#background path  
INPUT_PATH = '/lustre/cmswork/dalsanto/Output_hzz4l/data.h5'

#random seed
seed=datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)
print('Random seed:'+str(seed))

total_epochs = args.epochs 
latentsize = args.latsize # number of nodes in each hidden layer
layers = args.layers #number of hidden layers
weight_clipping = args.weight_clipping

patience = 2000 # number of epochs between two consecutives saving points
nfile_REF = 1 #number of files in REF repository
nfile_SIG = 1 #number of files in SIG repository

#output path
OUTPUT_PATH = args.output 
ID = '/Higgs_patience'+str(patience)+'_ref'+str(args.frac_ref)+'_sig'+str(args.signal)+'_epochs'+str(total_epochs)+'_latent'+str(latentsize)+'_layers'+str(layers)+'_wclip'+str(weight_clipping)+'_toys'+str(toy)+'_seed'+str(seed) #'_bkg'+str(args.frac_bkg)+'_sig'+str(args.frac_sig)+
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

#check if the toy label has already been used. If it is so, exit the program without repeating.
# if os.path.isfile(f"{OUTPUT_PATH}_t.txt"):
#     exit()

print('Start analyzing Toy '+toy)

def double_logger(stringa):
    print(stringa)
    with open(f"{OUTPUT_PATH}/check_seed{seed}.txt", "a") as norm_file:
        norm_file.write(str(stringa+'\n'))
        
#---------------------------------------
start_time=time.time()

#Read Data

#BACKGROUND+REFERENCE

dfHiggs = BuildSample_DY_Higgs(FILE_NAME=INPUT_PATH, seed=seed, nfiles=nfile_REF)
# uno per il reference e uno per i dati 

Frac_ref = args.frac_ref
'''
Frac_bkg = args.frac_bkg
Frac_sig = args.frac_sig
'''

selected_columns=['pt1', 'pt2', 'pt3', 'pt4', 'eta1', 'eta2', 'eta3', 'eta4', 'phi1', 'phi2', 'phi3', 'phi4']#['Z1_eta', 'Z1_mass', 'Z1_phi', 'Z1_pt', 'Z2_eta', 'Z2_mass', 'Z2_phi', 'Z2_pt'] #['H_mass']
# if len(selected_columns)==8:
#     dfHiggs['delta_phi'] = np.abs(dfHiggs['Z1_phi'] - dfHiggs['Z2_phi'])
#     dfHiggs.drop(columns=['Z1_phi','Z2_phi'])
#     selected_columns=['Z1_eta', 'Z1_mass', 'Z1_pt', 'Z2_eta', 'Z2_mass', 'Z2_pt', 'delta_phi']
#     print(dfHiggs.columns)
    
# if len(selected_columns)==12:
#     def smart_init(x,y):
#         a = x-y
#         a = np.where(a>0, a, 2*np.pi+a)
#         return a
#     dfHiggs['delta_phi21'] = smart_init(dfHiggs['phi2'], dfHiggs['phi1'])
#     dfHiggs['delta_phi31'] = smart_init(dfHiggs['phi3'], dfHiggs['phi1'])
#     dfHiggs['delta_phi41'] = smart_init(dfHiggs['phi4'], dfHiggs['phi1'])
#     dfHiggs.drop(columns=['phi1','phi2','phi3','phi4'])
#     selected_columns=['pt1', 'pt2', 'pt3', 'pt4', 'eta1', 'eta2', 'eta3', 'eta4', 'delta_phi21', 'delta_phi31', 'delta_phi41']
#     print(dfHiggs.columns)

# #select the relevant variable
# mask = (dfHiggs['pt1']>20) & (dfHiggs['pt2']>10) & (dfHiggs['pt3']>10) & (dfHiggs['H_mass']<300) & (dfHiggs['H_mass']>100)

#extract the events with the correct statistics
# N_bkg_2_p = np.random.poisson(lam=np.sum(dfHiggs['weights'][(dfHiggs['process']==2) & mask]), size=1)[0]
# N_bkg_3_p = np.random.poisson(lam=np.sum(dfHiggs['weights'][(dfHiggs['process']==3) & mask]), size=1)[0]
# N_Bkg_p = N_bkg_2_p + N_bkg_3_p
# double_logger(f'N_bkg_2 and N_bkg_3: {N_bkg_2_p} , {N_bkg_3_p}')

# N_ref_start_2 = len(dfHiggs['H_mass'][(dfHiggs['process']==2) & mask])
# N_ref_2 = np.min([N_ref_start_2-N_bkg_2_p, int(N_ref_start_2 * Frac_ref)])
# N_ref_start_3 = len(dfHiggs['H_mass'][(dfHiggs['process']==3) & mask])
# N_ref_3 = np.min([N_ref_start_3-N_bkg_3_p, int(N_ref_start_3 * Frac_ref)])
# N_ref = N_ref_2+N_ref_3
N_ref = len(...) # completare

# N_sig_start_4 = len(dfHiggs['H_mass'][(dfHiggs['process']==4) & mask])
# N_sig_start_5 = len(dfHiggs['H_mass'][(dfHiggs['process']==5) & mask])
# if args.signal=='True':
#     N_sig_4_p = np.random.poisson(lam=np.sum(dfHiggs['weights'][(dfHiggs['process']==4) & mask]), size=1)[0]
#     N_sig_5_p = np.random.poisson(lam=np.sum(dfHiggs['weights'][(dfHiggs['process']==5) & mask]), size=1)[0]
#     N_Sig_p = N_sig_4_p + N_sig_5_p
#     double_logger(f'N_sig_2 and N_sig_3: {N_sig_4_p} , {N_sig_5_p}')
# elif args.signal!='True':
#     N_sig_4_p, N_sig_5_p = np.random.poisson(lam=0, size=1)[0], np.random.poisson(lam=0, size=1)[0]
#     N_sig_4_p, N_sig_5_p, N_Sig_p = 0, 0, 0

    
#REFERENCE
HLF_REF_2 = dfHiggs[selected_columns][(dfHiggs['process']==2) & mask][:N_ref_2] # prendo tutto
# weights_REF_2 = dfHiggs['weights'][(dfHiggs['process']==2) & mask][:N_ref_2] *  N_ref_start_2/N_ref_2
# HLF_REF_3 = dfHiggs[selected_columns][(dfHiggs['process']==3) & mask][:N_ref_3]
# weights_REF_3 = dfHiggs['weights'][(dfHiggs['process']==3) & mask][:N_ref_3] *  N_ref_start_3/N_ref_3
# HLF_REF = np.append(HLF_REF_2,HLF_REF_3,axis=0)
# weights_REF = np.append(weights_REF_2, weights_REF_3,axis=0)
#BACKGROUND
HLF_BKG_1 = dfHiggs[selected_columns][(dfHiggs['process']==1) & mask]  # prendo tutto
# weights_BKG_1 = dfHiggs['weights'][(dfHiggs['process']==1) & mask]
# HLF_BKG_2 = dfHiggs[selected_columns][(dfHiggs['process']==2) & mask][N_ref_2:N_ref_2+N_bkg_2_p]
# weights_BKG_2 = dfHiggs['weights'][(dfHiggs['process']==2) & mask][N_ref_2:N_ref_2+N_bkg_2_p] * N_ref_start_2/N_bkg_2_p
# HLF_BKG_3 = dfHiggs[selected_columns][(dfHiggs['process']==3) & mask][N_ref_3:N_ref_3+N_bkg_3_p]
# weights_BKG_3 = dfHiggs['weights'][(dfHiggs['process']==3) & mask][N_ref_3:N_ref_3+N_bkg_3_p] * N_ref_start_3/N_bkg_3_p
# if args.DY=='False':
#     HLF_BKG = np.append(HLF_BKG_2,HLF_BKG_3,axis=0)
#     weights_BKG = np.append(weights_BKG_2, weights_BKG_3,axis=0)
# elif args.DY=='True':
#     HLF_BKG = np.concatenate((HLF_BKG_1,HLF_BKG_2,HLF_BKG_3),axis=0)
#     weights_BKG = np.concatenate((weights_BKG_1, weights_BKG_2, weights_BKG_3),axis=0)
#     N_Bkg_p = N_Bkg_p + len(HLF_BKG_1)
#SIGNAL
HLF_SIG_4 = dfHiggs[selected_columns][(dfHiggs['process']==4) & mask][:N_sig_4_p]  # prendo tutto # può diventare lo spegnimento artificiale di un canale
# weights_SIG_4 = dfHiggs['weights'][(dfHiggs['process']==4) & mask][:N_sig_4_p] * N_sig_start_4/N_sig_4_p
# HLF_SIG_5 = dfHiggs[selected_columns][(dfHiggs['process']==5) & mask][:N_sig_5_p]
# weights_SIG_5 = dfHiggs['weights'][(dfHiggs['process']==5) & mask][:N_sig_5_p] * N_sig_start_5/N_sig_5_p
# HLF_SIG = np.append(HLF_SIG_4,HLF_SIG_5,axis=0)
# weights_SIG = np.append(weights_SIG_4, weights_SIG_5,axis=0)
# print(HLF_BKG.shape)
#DATA
# HLF_DATA = dfHiggs[selected_columns][(dfHiggs['process']==0) & mask]
# weights_DATA = dfHiggs['weights'][(dfHiggs['process']==0) & mask]
####################
# if 0==0:
#     double_logger('Subtracting DY+Jets from data')
#     ######################
#     '''
#     import pandas as pd
#     higgs_file_2 = h5py.File('/lustre/cmswork/dalsanto/Output_hzz4l_v2/data.h5','r')
#     dfH_2=pd.DataFrame({key:np.array(higgs_file_2.get(key)) for key in higgs_file_2.keys()})
#     dfH_2['delta_phi'] = np.abs(dfH_2['Z1_phi'] - dfH_2['Z2_phi'])
#     dfH_2.drop(columns=['Z1_phi','Z2_phi'])
#     print(dfH_2['delta_phi'][:10])
#     mask_2 = (dfH_2['pt1']>20) & (dfH_2['pt2']>10) & (dfH_2['pt3']>10) & (dfH_2['H_mass']<300) & (dfH_2['H_mass']>100)
#     HLF_BKG_1 = dfH_2[selected_columns][mask_2]
#     weights_BKG_1 = dfH_2['weights'][mask_2]
#     '''
#     ######################
#     HLF_DATA = np.append(HLF_DATA,HLF_BKG_1,axis=0)
#     weights_DATA = np.append(weights_DATA,-1*weights_BKG_1,axis=0)
# #####################

'''
#select the relevant variable
mask=(dfHiggs['H_mass']<300) & (dfHiggs['H_mass']>100) # & (dfHiggs['weights']>0)
mask_DY=(dfHiggs['process']==2) | (dfHiggs['process']==3)

N_Bkg_start = len(dfHiggs['H_mass'][(dfHiggs['labels']==0) & mask & mask_DY])
N_Sig_start = len(dfHiggs['H_mass'][(dfHiggs['labels']==2) & mask])
N_Bkg = int(N_Bkg_start * Frac_bkg)
N_Sig = int(N_Sig_start * Frac_sig)
N_Bkg_p = np.random.poisson(lam=N_Bkg, size=1)[0]
N_Sig_p = np.random.poisson(lam=N_Sig, size=1)[0]
N_ref =np.min([len(dfHiggs)-N_Bkg_p, int(N_Bkg_start * Frac_ref)]) #  #, 
double_logger('N_Bkg: '+str(N_Bkg)+' N_Bkg_Pois: '+str(N_Bkg_p))
double_logger('N_Sig: '+str(N_Sig)+' N_Sig_Pois: '+str(N_Sig_p))
double_logger('N_ref: '+str(N_ref))

#REFERENCE
HLF_REF = dfHiggs['H_mass'][(dfHiggs['labels']==0) & mask & mask_DY][:N_ref]
weights_REF = dfHiggs['weights'][(dfHiggs['labels']==0) & mask & mask_DY][:N_ref] *  N_Bkg_start/N_ref
#BACKGROUND
HLF_BKG = dfHiggs['H_mass'][(dfHiggs['labels']==0) & mask & mask_DY][N_ref:N_ref+N_Bkg_p]
weights_BKG = dfHiggs['weights'][(dfHiggs['labels']==0) & mask & mask_DY][N_ref:N_ref+N_Bkg_p] * N_Bkg_start/N_Bkg_p
print(f"Bkg weights sum: {np.sum(weights_BKG)} , lenght bkg {len(HLF_BKG)}")
#SIGNAL
HLF_SIG = dfHiggs['H_mass'][(dfHiggs['labels']==2) & mask][:N_Sig_p]
weights_SIG = dfHiggs['weights'][(dfHiggs['labels']==2) & mask][:N_Sig_p] * N_Sig_start/N_Sig_p#*5
print(f"Sig weights sum: {np.sum(weights_SIG)} , lenght sig {len(HLF_SIG)}")
'''



'''
plot prima 
fig2=plt.figure(figsize=(9,9))
frame1=fig2.add_axes((.1,.55,.8,.35))
plt.hist(HLF_REF,bins=50,range=(100,300),weights=weights_REF,label='Reference',log=False, alpha=0.75, edgecolor='black', linewidth=1.2)
plt.legend(loc='best')
plt.grid(True,linestyle=':')
#frame1.set_xticklabels([]) 
frame2=fig2.add_axes((.1,.1,.8,.35)) 
plt.hist([HLF_BKG,HLF_SIG],bins=50,range=(100,300),weights=[weights_BKG,weights_SIG],stacked=True,label=['Bkg','Signal'],log=False, alpha=0.75, edgecolor='black', linewidth=1.2)
plt.legend(loc='best')
plt.grid(True,linestyle=':')
plt.savefig(OUTPUT_PATH+ID+'_histo_density.png')
'''

#TARGETS
if args.CMS=='False':
    target_REF = np.zeros(N_ref)
    target_DATA = np.ones(N_Bkg_p+N_Sig_p)
    target = np.append(target_REF, target_DATA)
    target = np.expand_dims(target, axis=1)
#     weights = np.concatenate((weights_REF, weights_BKG, weights_SIG), axis=0)
#     weights = np.expand_dims(weights, axis=1)
    feature = np.concatenate((HLF_REF, HLF_BKG, #HLF_SIG), axis=0)
    #feature = np.expand_dims(feature, axis=2)
    feature = np.concatenate((feature, weights, target), axis=1) # tolgo i weights
if args.CMS=='True':
    double_logger('Using CMS data!!!')
    target_REF = np.zeros(N_ref)
    target_DATA = np.ones(len(HLF_DATA))
    target = np.append(target_REF, target_DATA)
    target = np.expand_dims(target, axis=1)
    weights = np.concatenate((weights_REF*np.sum(weights_DATA)/np.sum(dfHiggs['weights'][((dfHiggs['labels']==0) | (dfHiggs['labels']==2)) & mask]), weights_DATA), axis=0)
    weights = np.expand_dims(weights, axis=1)
    feature = np.concatenate((HLF_REF, HLF_DATA), axis=0)
    #feature = np.expand_dims(feature, axis=1)
    feature = np.concatenate((feature, weights, target), axis=1)
np.random.shuffle(feature)
print(f'feature shape: {feature.shape}')
weights = feature[:, -2] # occupanze
target = feature[:, -1] 
feature = feature[:, :-2]

#remove MASS from the input features
#mass = feature[:, -1]
#feature = feature[:, :-1]

#standardize dataset

for j in range(feature.shape[1]):
    vec = feature[:, j]
    mean = np.mean(vec)
    std = np.std(vec)
    if np.min(vec) < 0:
        #save the normalization parameters
        with open(OUTPUT_PATH+ID+'_normalization.txt','a+') as norm_file:
            norm_file.write(str(f'Standardization -> Mean: {mean}, Std: {std}; '))
        vec = vec-mean
        vec = vec/std
    elif np.max(vec) > 1.0:# Assume data is exponential -- just set mean to 1.
        #save the normalization parameters
        with open(OUTPUT_PATH+ID+'_normalization.txt','a+') as norm_file:
            norm_file.write(str(f'Standardization expo -> Mean: {mean}; '))
        vec = vec *1./ mean
    feature[:, j] = vec
    

double_logger(f'Target: {target.shape}')
double_logger(f'Features: {feature.shape}')
print('Start training Toy '+toy)
#--------------------------------------------------------------------

#Loss function definition                                                                                                                                                                                             
def LossWeighted(yTrue, yPred, w):
    return K.sum( (1-yTrue)*w*(K.exp(yPred)-1) - yTrue*w*yPred) #N_D/float(N_R)

'''#Define the callback
class myCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % int(total_epochs/20) == 0:
            print(f"Epoch: {epoch}, Logs:" ,logs)
            with open(f"{OUTPUT_PATH}/check.txt", "a") as myfile:
                myfile.write(f"Epoch: {epoch}, Logs: {logs}\n")
'''
def training_epoch(train_x,train_y,w):
    with tf.GradientTape() as tape:
        predicted_y = tf.squeeze(BSMfinder(train_x, training=True))
        loss_value = LossWeighted(train_y,predicted_y,weights)
    gradients = tape.gradient(loss_value, BSMfinder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, BSMfinder.trainable_variables))
    return loss_value.numpy()
    
# training
BSMfinder = NPL_Model(feature.shape[1], latentsize, layers, weight_clipping, internal_activation)
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
best_loss=10**5
best_loss_epoch=0
for epoch_num in range(total_epochs):
    if epoch_num==0: time_zero = time.time()
    for (train_x,train_y,w) in batched_dataset:
        loss_value = training_epoch(train_x,train_y,w)
        loss_list.append(loss_value)
        if loss_value<best_loss:
            best_loss=loss_value
            best_loss_epoch=epoch_num
    if best_loss_epoch<epoch_num-15000: break
    if epoch_num==10: double_logger(f'Time for 1 training epoch {(time.time()-time_zero)/epoch_num:.3f}')
    if epoch_num%patience==0: double_logger(f'Epoch {epoch_num}, Loss {loss_value:.4f}')

# test statistic                              
loss = np.array(loss_list)#np.array(hist.history['loss']) 
t_e_OBS = -2*loss[-1]

# save t   
log_t = OUTPUT_PATH+ID+'_t.txt'
out = open(log_t,'w')
out.write("%f\n" %(t_e_OBS))
out.close()
'''
Data = feature[target==1]
Reference = feature[target!=1]
print(Data.shape, Reference.shape)               
# inference
f_Data = BSMfinder(Data, training=False)
f_Reference = BSMfinder(Reference, training=False)
f_All = BSMfinder(feature, training=False)
'''
if feature.shape[1]==1:
    bool_plotter=True
    with open(OUTPUT_PATH+ID+'_normalization.txt', "r") as f:
        normalizzazione = f.readline()
        if 'expo' not in normalizzazione:
            print("WRONG NORMALIZATION APPLIED FOR THE PLOT!")
            bool_plotter=False
        normalizzazione = float(normalizzazione.split("Mean: ")[1].split(";")[0])

    
    if bool_plotter==True:
        fig2=plt.figure(figsize=(9,9))
        frame1=fig2.add_axes((.1,.55,.8,.35))
        plt.hist(HLF_REF,bins=50,range=(100,300),weights=weights_REF,label='Reference',log=False, alpha=0.75, edgecolor='black', linewidth=1.2)
        plt.hist([HLF_BKG,HLF_SIG],bins=50,range=(100,300),weights=[weights_BKG,weights_SIG],stacked=True,label=['Bkg','Signal'],log=False, alpha=0.75, edgecolor='black', linewidth=1.2)
        plt.legend(loc='best')
        plt.grid(True,linestyle=':')
        #frame1.set_xticklabels([]) 
        frame2=fig2.add_axes((.1,.1,.8,.35)) 
        x_step = np.expand_dims(np.linspace(np.min(feature.flatten()),np.max(feature.flatten()),10000), axis=1)
        plt.plot(x_step.flatten()*normalizzazione,np.exp(BSMfinder(x_step, training=False)),label='Reference neural network output') # plot output della rete -> 
        plt.legend(loc='best')
        plt.grid(True,linestyle=':')
        plt.savefig(OUTPUT_PATH+ID+'_1dratio.png')
        
        #plt.figure(figsize=(6,5))
        #print(np.array(BSMfinder(feature, training=False)).shape)
        #plt.hist(np.exp(BSMfinder(feature, training=False)),bins=50,label='Activation distribution',log=False, alpha=0.75, edgecolor='black', linewidth=1.2)
        #plt.legend(loc='best')
        #plt.grid(True,linestyle=':')
        #plt.savefig(OUTPUT_PATH+ID+'_activation.png')
        #x_step = np.expand_dims(np.linspace(np.min(feature.flatten()),np.max(feature.flatten()),10000), axis=1)
        #plt.plot(x_step.flatten()*normalizzazione,np.exp(BSMfinder(x_step, training=False)),label='Reference neural network output')
        #plt.legend(loc='best')
        #plt.grid(True,linestyle=':')
        #plt.savefig(OUTPUT_PATH+ID+'_1dratio.png')
        
# write the loss history                                                                                                     
log_history = OUTPUT_PATH+ID+'_history.h5'
f = h5py.File(log_history,"w")
keepEpoch = np.array(range(total_epochs))
keepEpoch = keepEpoch % patience == 0
f.create_dataset('loss', data=loss, compression='gzip')
f.close()

# save the model                                                                                         

log_model = OUTPUT_PATH+ID+'_model.json'
log_weights = OUTPUT_PATH+ID+'_weights.h5'
model_json = BSMfinder.to_json()
with open(log_model, "w") as json_file:
    json_file.write(model_json)
BSMfinder.save_weights(log_weights)



# save outputs 

#log_ratio = OUTPUT_PATH+ID+'_reco_vs_ref.h5'
#f = h5py.File(log_ratio,"w")
#f.close()

double_logger('Output saved for Toy '+toy)
double_logger("Execution time: ", time.time()-start_time)
double_logger('----------------------------\n')


'''



bin_intervals=[0,50,75,100,125,150,175,200,225,250,300,350,500]
ref_mass = [x for x, label in zip(mass,target) if label==0] 
data_mass = [x for x, label in zip(mass,target) if label==1]

bin_ref_baricenter, bin_ref_baricenter_minus, bin_ref_baricenter_plus = [], [], []
for i in range(len(bin_intervals)-1):
    lista = None
    lista = [x for x in ref_mass if (x>=bin_intervals[i]) and (x<bin_intervals[i+1])] 
    bin_ref_baricenter.append(np.median(lista))
    bin_ref_baricenter_minus.append(np.median(lista) - np.quantile(lista, 0.15865))
    bin_ref_baricenter_plus.append(np.quantile(lista, 0.84135)-np.median(lista))
bin_ref_baricenter = np.array(bin_ref_baricenter)
bin_ref_baricenter_minus = np.array(bin_ref_baricenter_minus)
bin_ref_baricenter_plus = np.array(bin_ref_baricenter_plus)
bin_ref_baricenter_err = np.array([bin_ref_baricenter_minus, bin_ref_baricenter_plus])

ref_mass_hist=np.histogram(ref_mass,bin_intervals)[0]
ref_mass_hist_err = np.array([np.sqrt(x*(1-x/len(ref_mass))) for x in ref_mass_hist])
data_mass_hist=np.histogram(data_mass,bin_intervals)[0]*N_ref/(N_Bkg_p+N_Sig_p)
data_mass_hist_err = np.array([np.sqrt(x*(1-x/(len(data_mass)*N_ref/(N_Bkg_p+N_Sig_p)))) for x in data_mass_hist])
density_ratio = np.exp(f_Reference)

reco_mass_hist, reco_mass_hist_err = [], []
for i in range(len(bin_intervals)-1):
    lista = None
    lista = np.array([x for x,y in zip(density_ratio, ref_mass) if (y>=bin_intervals[i]) and (y<bin_intervals[i+1])])
    reco_mass_hist.append(np.sum(lista))
reco_mass_hist = np.array(reco_mass_hist)
reco_mass_hist_err = np.array([np.sqrt(x*(1-x/np.sum(reco_mass_hist))) for x in reco_mass_hist])
f.create_dataset('bin_ref_baricenter', data=bin_ref_baricenter, compression='gzip')
f.create_dataset('bin_ref_baricenter_std', data=bin_ref_baricenter_err, compression='gzip') 
#f.create_dataset('feature', data=feature, compression='gzip')
f.create_dataset('data_mass_hist', data=data_mass_hist, compression='gzip')
f.create_dataset('data_mass_hist_err', data=data_mass_hist_err, compression='gzip')
f.create_dataset('ref_mass_hist', data=ref_mass_hist, compression='gzip')
f.create_dataset('ref_mass_hist_err', data=ref_mass_hist_err, compression='gzip')
f.create_dataset('reco_mass_hist', data=reco_mass_hist, compression='gzip')
f.create_dataset('reco_mass_hist_err', data=reco_mass_hist_err, compression='gzip')
#f.create_dataset('target', data=target, compression='gzip')
f.close()



log_predictions =OUTPUT_PATH+OUTPUT_FILE_ID+'_predictions.h5'
f = h5py.File(log_predictions,"w")
#mass_sorter=np.argsort(mass)
#mass=mass[mass_sorter]
#f_All=f_All[mass_sorter]

mass_down=[np.mean([x for x in mass if (x>=i) and (x<i+2)]) for i in range(0,500,2)]
like_ratio_f_All = np.exp(f_All)
like_ratio_f_All_down=[np.mean([y for x,y in zip(mass,like_ratio_f_All) if (x>=i) and (x<i+2)]) for i in range(0,500,2)]
mass_down=[x for (i,x) in enumerate(mass_down) if np.logical_not(np.isnan(mass_down))[i]]
like_ratio_f_All_down=[x for (i,x) in enumerate(like_ratio_f_All_down) if np.logical_not(np.isnan(like_ratio_f_All_down))[i]]
f.create_dataset('mass', data=mass_down, compression='gzip')
#f.create_dataset('feature', data=feature, compression='gzip')
f.create_dataset('net_output', data=like_ratio_f_All_down, compression='gzip')
#f.create_dataset('target', data=target, compression='gzip')
f.close()
'''





















'''
binned_likelihood_ratio, binned_likelihood_ratio_minussigma, binned_likelihood_ratio_plussigma = [], [], []
for i in range(len(bin_intervals)-1):
    lista=None
    lista=[x for x, mass_index in zip(density_ratio, ref_mass) if (mass_index>=bin_intervals[i]) and (mass_index<bin_intervals[i+1])]
    binned_density_ratio.append(np.median(lista))
    binned_density_ratio_minussigma.append(np.median(lista) - np.quantile(lista, 0.15865))
    binned_density_ratio_plussigma.append(np.quantile(lista, 0.84135) - np.mediant(lista))
binned_density_ratio = np.array(binned_density_ratio)
binned_density_ratio_minussigma = np.array(binned_density_ratio_minussigma)
binned_density_ratio_plussigma = np.array(binned_density_ratio_plussigma)
'''
#binned_likelihood_ratio_err = np.array([binned_likelihood_ratio_minussigma, binned_likelihood_ratio_plussigma])
#print(ref_mass_hist)
#print(binned_likelihood_ratio)

'''
reco_mass_hist = ref_mass_hist * binned_density_ratio
reco_mass_hist_minus = np.sqrt((ref_mass_hist_err * binned_likelihood_ratio)**2 + (ref_mass_hist * binned_likelihood_ratio_minussigma)**2)
reco_mass_hist_plus = np.sqrt((ref_mass_hist_err * binned_likelihood_ratio)**2 + (ref_mass_hist * binned_likelihood_ratio_plussigma)**2)
reco_mass_hist_err = np.array([reco_mass_hist_minus, reco_mass_hist_plus])
'''
#ratio_data_ref = data_mass_hist/ref_mass_hist
#ratio_data_ref_err = np.sqrt((data_mass_hist_err/ref_mass_hist)**2+(ref_mass_hist_err*data_mass_hist/ref_mass_hist**2)**2)

#mass_down=[np.mean([x for x in mass if (x>=i) and (x<i+1)]) for i in range(0,300,1)]
#f_All_down=[np.mean([y for x,y in zip(mass,f_All) if (x>=i) and (x<i+1)]) for i in range(0,300,1)]
#mass_down=[x for (i,x) in enumerate(mass_down) if np.logical_not(np.isnan(mass_down))[i]]
#f_All_down=[x for (i,x) in enumerate(f_All_down) if np.logical_not(np.isnan(f_All_down))[i]]





