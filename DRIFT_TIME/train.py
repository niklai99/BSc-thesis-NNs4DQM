import numpy as np
import os
import argparse
from keras import callbacks
import h5py
import keras.backend as K

from DataReader import DataReader
from ModelBuilder import ModelBuilder


class myCallback(callbacks.Callback):
    '''gestisce il callback quando finisce una epoch'''
    
    
    def __init__(self, epochs: int, output_path: str):
        self.epochs = epochs
        self.output_path = output_path
      
    def on_epoch_end(self, epoch, logs=None):
        '''quando finisce una epoch controlla che sia un divisore particolare e scrive sul file un check'''
        
        if epoch % int(self.epochs/50) == 0:
            with open(f"{self.output_path}/check.txt", "a") as myfile:
                myfile.write(f"Epoch: {epoch}, Logs: {logs}\n")
                

                
def argParser():
    '''gestisce gli argomenti passati da linea di comando'''
    
    parser = argparse.ArgumentParser()    #Python tool that makes it easy to create an user-friendly command-line interface
    parser.add_argument('-o','--output', type=str, help="output directory", required=True)
    parser.add_argument('-t','--toys', type=str, default=40, help="number of repetitions", required=False)
    parser.add_argument('-sig','--signal', type=int, default=0, help="number of signal events", required=False)
    parser.add_argument('-bkg','--background', type=int, default=4000, help="number of background events", required=False)
    parser.add_argument('-ref','--reference', type=int, default=40000, help="number of reference events", required=False)
    parser.add_argument('-epochs','--epochs', type=int, default=100000, help="number of epochs", required=False)
    parser.add_argument('-latsize','--latsize', type=int, default=3, help="number of nodes in each hidden layer", required=False)
    parser.add_argument('-layers','--layers', type=int, default=1, help="number of layers", required=False)
    parser.add_argument('-wclip','--weight_clipping', type=float, default=7, help="weight clipping", required=False)
    parser.add_argument('-patience','--patience', type=int, default=1000, help="number of epochs between two consecutives saving points", required=False)
    
    return parser.parse_args()


def poisson_fluctuation(N_Data: int):
    '''calcola la fluttuazione poissoniana'''
    
    return np.random.poisson(lam=N_Data, size=1)[0]


def make_output_path(output_path: str, label: str):
    '''unisce il percorso di output con la label delle cartelle e file'''
    
    return output_path + label

    
def read_data(file_name: str, n_data: int):
    '''legge la distribuzione da un file'''
    
    data_instance = DataReader()
    data_df = data_instance.build_sample(file_name, n_data)
    
    return data_df


def make_target(n_reference: int, n_background: int, n_signal: int):
    '''costruisce i targets'''
    
    ref_target = np.zeros(n_reference)
    data_target = np.ones(n_background+n_signal)
    target = np.append(ref_target, data_target)
    target = np.expand_dims(target, axis=1)
    
    return target


def make_feature(ref_df, data_df, target):
    '''costruisce la feature'''
    
    feature = np.concatenate((ref_df, data_df), axis=0)
    feature = np.expand_dims(feature, axis=1)
    feature = np.concatenate((feature, target), axis=1)
    np.random.shuffle(feature)
    
    return feature


def normalize_dataset(feature, output_path: str, label: str):
    '''normalizza i dati'''
    
    for j in range(feature.shape[1]):
        vec = feature[:, j]
        mean = np.mean(vec)
        std = np.std(vec)
        if np.min(vec) < 0:
            # save the normalization parameters
            with open(output_path+label+'_normWalization.txt','w') as norm_file:
                norm_file.write(str(f'Standardization -> Mean: {mean}, Std: {std}'))
            vec = vec-mean
            vec = vec/std
        elif np.max(vec) > 1.0:# Assume data is exponential -- just set mean to 1.
            # save the normalization parameters
            with open(output_path+label+'_normalization.txt','w') as norm_file:
                norm_file.write(str(f'Exponential standardization -> Mean: {mean}'))
            vec = vec *1./ mean
        feature[:, j] = vec
        
    return feature


def custom_loss(N_R: float, N_D: float):
    '''permette di avere parametri nella loss function'''
    
    def Loss(yTrue, yPred):
        return K.sum(-1.*yTrue*(yPred) + (1-yTrue)*N_D/float(N_R)*(K.exp(yPred)-1))
    
    return Loss



def main(args):

    # output path
    OUTPUT_PATH = args.output

    # number of toy samples
    N_TOYS = args.toys
    
    N_Sig = args.signal          # set this to 0 for training on reference
    N_Bkg = args.background      # usually 10'000
    N_Data = N_Bkg + N_Sig       # total number of data expected
    N_Ref = args.reference       # usually 200'000
    
    EPOCHS = args.epochs         # number of epochs (a lot)
    LATENT_SIZE = args.latsize   # number of nodes in each hidden layer (what is this?)
    LAYERS = args.layers         # number of hidden layers
    
    PATIENCE = args.patience     # number of epochs between two consecutives saving points
    
    # NN restriction
    WEIGHT_CLIPPING = args.weight_clipping
    
    LABEL = (
        '/E'+str(EPOCHS)+'_latent'+str(LATENT_SIZE)+'_layers'+str(LAYERS)+'_wclip'+str(WEIGHT_CLIPPING)
        +'_ntoy'+str(N_TOYS)+'_ref'+str(N_Ref)+'_bkg'+str(N_Bkg)+'_sig'+str(N_Sig)+'_patience'+str(PATIENCE)
    )
    
    OUTPUT_PATH = make_output_path(OUTPUT_PATH, LABEL)
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    
    # SIGNAL POISSON FLUCTUATIONS
    N_Sig_p = poisson_fluctuation(N_Sig)
#     print('N_Sig: '+str(N_Sig))
#     print('N_Sig_Pois: '+str(N_Sig_p))

    # BACKGROUND POISSON FLUCTUATIONS
    N_Bkg_p = poisson_fluctuation(N_Bkg)
#     print('N_Bkg: '+str(N_Bkg))
#     print('N_Bkg_Pois: '+str(N_Bkg_p))
    

    # build reference and data dataframes
#     REF_DF = build_data(n_background=N_Ref, n_signal=0)
#     DATA_DF = build_data(n_background=N_Bkg_p, n_signal=N_Sig_p)
    REF_DF = read_data(file_name='RUN1252.txt', n_data=N_Ref)
    DATA_DF = read_data(file_name='RUN1252.txt', n_data=N_Bkg_p)

    # create target and features
    target = make_target(N_Ref, N_Bkg_p, N_Sig_p)
    feature = make_feature(REF_DF, DATA_DF, target)
    
    # select target and features
    target = feature[:, -1]
    feature = feature[:, :-1]
   
    # normalizzazione
    feature = normalize_dataset(feature, OUTPUT_PATH, LABEL)
    
    # training
#     batch_size = feature.shape[0]
#     BSMfinder = NPL_Model_v1(feature.shape[1], LATENT_SIZE, LAYERS, WEIGHT_CLIPPING)
#     BSMfinder.compile(loss = custom_loss(N_Ref, N_Data),  optimizer = 'adam')
#     hist = BSMfinder.fit(feature, target, batch_size=batch_size, epochs=EPOCHS, verbose=0, callbacks=[myCallback(EPOCHS, OUTPUT_PATH)],)
    
    batch_size = feature.shape[0]
    n_inputs = feature.shape[1]
    
    NPLModel = ModelBuilder(
                n_input=n_inputs,
                latentsize=LATENT_SIZE,
                layers=LAYERS,
                weight_clipping=WEIGHT_CLIPPING,
                internal_activation='tanh',        # usare tanh, sigmoid non va molto bene
                batch_norm_bool=True,              # mette un batch_normalization layer tra input e hidden layers
                more_batch_norm_bool=True,         # mette un batch_normalization layer tra gli hidden
                custom_activation_bool=True,       # usa una custom activation per l'output, altrimenti linear
                custom_const=1                     # parametro della custom activation function 
            )
    
    BSMfinder = NPLModel()
    BSMfinder.compile(loss = custom_loss(N_Ref, N_Data),  optimizer = 'adam')
    
    hist = BSMfinder.fit(
                feature, target, 
                batch_size=batch_size, epochs=EPOCHS, 
                verbose=0, 
                callbacks=[myCallback(EPOCHS, OUTPUT_PATH)]
            )
    
    
    
    # metrics                                   
    loss = np.array(hist.history['loss'])
    
    # test statistic                                   
    final_loss = loss[-1]
    t_e_OBS = -2*final_loss
    
    # save t                                           
    log_t = OUTPUT_PATH+LABEL+'_t.txt'
    out = open(log_t,'w')
    out.write("%f\n" %(t_e_OBS))
    out.close()

    # write the loss history    
    log_history = OUTPUT_PATH+LABEL+'_history'+str(PATIENCE)+'.h5'
    f = h5py.File(log_history,"w")
    keepEpoch = np.array(range(EPOCHS))
    keepEpoch = keepEpoch % PATIENCE == 0
    f.create_dataset('loss', data=loss[keepEpoch], compression='gzip')
    f.close()

    # save the model                                           
    log_model = OUTPUT_PATH+LABEL+'_model.json'
    log_weights = OUTPUT_PATH+LABEL+'_weights.h5'
    model_json = BSMfinder.to_json()
    with open(log_model, "w") as json_file:
        json_file.write(model_json)
    BSMfinder.save_weights(log_weights)
    
    
    return


if __name__ == "__main__":
    args = argParser()
    main(args)