import os

class RunParameters:
    '''classe che gestisce i parametri della run'''
    
    def __init__(self, out_dir: str, ndir=0, ntoy=40, magic=0):
        '''init: imposto il nome della cartella di output, genero il nome della cartella contenente il file e cerco il file'''
        
        # output directory
        self.out_dir = out_dir
        # entro nella output directory e dentro alla cartella con il numero magico del toy
        # prendo come folder_name il nome della cartella indicizzata da ndir
        self.folder_name = os.listdir(self.out_dir+f'/{magic}/')[ndir]
        # costruisco il nome del file contenente i t
        self.fname = [name for name in os.listdir(self.out_dir+f'/{magic}/'+self.folder_name) if name.endswith('_t.txt')][0]
        
        return
    
    def fetch_parameters(self) -> list:
        '''uso il nome del file e della cartella per estrarre i parametri della run'''
        
        # estrazione dei parametri della run da fname e folder_name
        self.epochs = int( (self.folder_name.split("E", 1)[1]).split('_')[0] )
        self.latent = int( (self.folder_name.split("latent", 1)[1]).split("_", 1)[0] )
        self.layers = int( (self.folder_name.split("layers", 1)[1]).split("_", 1)[0] )
        self.w_clip = (self.folder_name.split("wclip", 1)[1]).split("_", 1)[0]
        self.toys = int( (self.folder_name.split("toy", 1)[1]).split("_", 1)[0] )
        self.ref = int( (self.folder_name.split("ref", 1)[1]).split("_", 1)[0] )
        self.bkg = int( (self.folder_name.split("bkg", 1)[1]).split("_", 1)[0] )
        self.sig = int( (self.folder_name.split("sig", 1)[1]).split("_", 1)[0] )
        self.check_point_t = int( (self.folder_name.split("patience", 1)[1]).split("_", 1)[0] )
  
        # impacchetto i parametri in una lista da ritornare
        self.parameters = [self.toys, self.w_clip, self.epochs, self.check_point_t, self.ref, self.bkg, self.sig, self.latent, self.layers]
        
        return self.parameters
    
    def print_parameters(self):
        '''stampo i parametri per controllare corrispondano alla run'''
        
        print('\nFolder name: ' + self.folder_name)
        print('File name: ' + self.fname)
        
        print(f'\nParameters:                                              \
                        \n Toys:          {self.toys}                      \
                        \n Latent space:  {self.latent}                    \
                        \n Layers:        {self.layers}                    \
                        \n W_clip:        {self.w_clip}                    \
                        \n Epochs:        {self.epochs}                    \
                        \n Patience:      {self.check_point_t}             \
                        \n Ref, Bkg, Sig: {self.ref} {self.bkg} {self.sig} \n'
             )
        
        return  
    
    def fetch_file(self) -> str:
        '''genero il nome completo del file contenente il t finale'''
        
        self.tfile = (
                        '/E'+str(self.epochs)+'_latent'+str(self.latent)+'_layers'+str(self.layers)
                        +'_wclip'+str(self.w_clip)
                        +'_ntoy'+str(self.toys)+'_ref'+str(self.ref)+'_bkg'+str(self.bkg)+'_sig'+str(self.sig)
                        +'_patience'+str(self.check_point_t)+'_t.txt'
        )
        
        # tolgo gli spazi vuoti che vengono automaticamente inseriti andando a capo con \
        self.tfile = self.tfile.replace(' ', '')
        
        return self.tfile
    
    def fetch_history(self) -> str:
        '''genero il nome completo del file contenente il t per ogni checkpoint'''
        
        self.thistory = (
                        '/E'+str(self.epochs)+'_latent'+str(self.latent)+'_layers'+str(self.layers)
                        +'_wclip'+str(self.w_clip)
                        +'_ntoy'+str(self.toys)+'_ref'+str(self.ref)+'_bkg'+str(self.bkg)+'_sig'+str(self.sig)
                        +'_patience'+str(self.check_point_t)+'_history'+str(self.check_point_t)+'.h5'
        )
        
        # tolgo gli spazi vuoti che vengono automaticamente inseriti andando a capo con \
        self.thistory = self.thistory.replace(' ', '')
        
        return self.thistory