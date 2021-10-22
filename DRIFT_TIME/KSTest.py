import numpy as np
from scipy.stats import chi2

def KSDistanceToUniform(sample):
    Ntrials      = sample.shape[0]
    ECDF         = np.array([i*1./Ntrials for i in np.arange(Ntrials+1)])
    sortedsample = np.sort(sample)
    sortedsample = np.append(0, np.sort(sample))
    KSdist       = 0
    if Ntrials==1:
        return np.maximum(1-sortedsample[1], sortedsample[1])
    else:
        return np.max(
            [
                np.maximum(
                    np.abs(sortedsample[i + 1] - ECDF[i]),
                    np.abs(sortedsample[i + 1] - ECDF[i + 1]),
                )
                for i in np.arange(Ntrials)
            ]
        )

def KSTestStat(data, ndof):
    sample = chi2.cdf(data, ndof)
    return KSDistanceToUniform(sample)

def GenUniformToy(Ntrials):
    sample = np.random.uniform(size=(Ntrials,))
    return KSDistanceToUniform(sample)

def GetTSDistribution(Ntrials, Ntoys=1000):
    KSdistDistribution = []
    for _ in range(Ntoys):
        KSdist = GenUniformToy(Ntrials)
        KSdistDistribution.append(KSdist)
    return np.array(KSdistDistribution)

def pvalue(KSTestStat_Value, KSdistDistribution):
    return (
        np.sum(1 * (KSdistDistribution > KSTestStat_Value))
        * 1.0
        / KSdistDistribution.shape[0]
    )

def GenToyFromEmpiricalPDF(sample):
    Ntrials = sample.shape[0]
    indeces = np.random.randint(low=0, high=Ntrials, size=(Ntrials,))
    return np.array([sample[indeces[i]] for i in range(Ntrials)])

def KS_test(sample, dof, Ntoys=100000):
    Ntrials            = sample.shape[0]
    KSTestStat_Value   = KSTestStat(sample, dof)
    KSdistDistribution = GetTSDistribution(Ntrials=Ntrials, Ntoys=Ntoys)
    return pvalue(KSTestStat_Value, KSdistDistribution)