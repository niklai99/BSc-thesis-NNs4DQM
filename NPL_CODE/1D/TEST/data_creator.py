import numpy as np
import pandas as pd
import scipy.stats

reference_len = 200000
background_len = 10000
signal_len = 50000

# reference
# reference = scipy.stats.expon.rvs(loc=0,scale=0.125,size=reference_len)
# df = pd.DataFrame({'feature0':reference})
# df.to_csv('ref.txt', index=False)


bkg = scipy.stats.expon.rvs(loc=0,scale=0.125, size=reference_len-signal_len)
sig = scipy.stats.norm.rvs(loc=0.8,scale=0.02, size=signal_len)
data = np.concatenate((np.array(bkg), np.array(sig)))
np.random.shuffle(data)

# df = pd.DataFrame({'feature0':bkg})
# df.to_csv('bkg.txt', index=False)

# df = pd.DataFrame({'feature0':sig})
# df.to_csv('sig.txt', index=False)

df = pd.DataFrame({'feature0':data})
df.to_csv('sig_bkg.txt', index=False)


