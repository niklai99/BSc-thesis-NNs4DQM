import numpy as np
import h5py
import scipy.stats

reference_len=500000
signal_len=10000


# hf = h5py.File('h0_ref1.h5','w')
# reference = scipy.stats.expon.rvs(loc=0,scale=0.125,size=reference_len)
# hf.create_dataset('mll', data=reference)
# hf.close()


hf = h5py.File('h1_ref.h5','w')
signal = scipy.stats.norm.rvs(loc=0.8,scale=0.02,size=signal_len)
hf.create_dataset('mll', data=signal)
hf.close()
