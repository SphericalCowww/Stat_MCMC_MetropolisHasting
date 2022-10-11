import sys, math
import re
import time
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from copy import deepcopy
from scipy import linalg

import warnings
warnings.filterwarnings("ignore");


SNUMBER = pow(10, -124);
def uniform(lowX, highX, x):
    xArr = np.array(x);
    if (xArr < lowX) or (highX < xArr):
        return SNUMBER;
    return np.ones_like(xArr)/(highX-lowX);
def gaussian(mu, sig, x):
    X = np.array(x);
    val = np.exp(-np.power(X-mu,2.0)/(2.0*np.power(sig,2.0)))\
         *(1.0/(sig*np.sqrt(2.0*np.pi)));
    val[val < SNUMBER] = SNUMBER;
    return val;

def priorFull(a, b):
    return lambda x : uniform(a, b, x);
def likelihoodFull(b, x):
    return lambda a : gaussian(a, b, x);
    

#data:  gaussian
#prior: uniform
#mcmc:  gaussian
def main():
    verbosity = 1;
    binN = 200;
    rangeX = [-10.0, 10.0];

    np.random.seed(2);
    dataMu = 1.0;
    dataSig = 1.0;
    dataN = 1000;

    mcmcMuMuInit = 1.0;
    mcmcMuSig = 1.0;
    mcmcN = 10000;
    mcmcNCut = int(mcmcN/10.0); #trimming
    thinF = 3;                  #thinning
    freezF = 1.0;               #annealing
#data
    nbins = np.linspace(rangeX[0], rangeX[1], binN+1)[:-1];
    dataPDF = np.random.normal(dataMu, dataSig, dataN);
    dataHist = np.zeros(binN);
    for x in dataPDF:
        if rangeX[0] < x and x < rangeX[1]:
            dataHist[int(binN*(x-rangeX[0])/(rangeX[1]-rangeX[0]))] += 1;
    priorHist = np.zeros(binN);
    for i, val in enumerate(nbins):
        priorHist[i] = uniform(rangeX[0], rangeX[1], val);
#MCMC
    iteration = np.linspace(0, mcmcN, mcmcN+1)[:-1];
    mcmcMuVal = np.copy(mcmcMuMuInit);
    mcmcMu = np.zeros(mcmcN);
    
    likelihood = likelihoodFull(dataSig, dataPDF);
    prior = priorFull(rangeX[0], rangeX[1]);

    iterAcc   = [];
    mcmcMuAcc = [];
    iterRej   = [];
    mcmcMuRej = [];
    for i in range(0, mcmcN):
        mcmcMuSigVal = mcmcMuSig/(1.0 + freezF*np.sqrt(i));   #anneal
        mcmcMuValNext = np.random.normal(mcmcMuVal, mcmcMuSigVal);
        likelihood1 = likelihood(mcmcMuVal);
        prior1      = prior(     mcmcMuVal);
        likelihood2 = likelihood(mcmcMuValNext);
        prior2      = prior(     mcmcMuValNext);
        L1 = np.sum(np.log(likelihood1));
        P1 = np.log(prior1);
        L2 = np.sum(np.log(likelihood2));
        P2 = np.log(prior2);
#Metropolis Hasting
        mcmcMu[i] = np.copy(mcmcMuVal);
        if (L2 + P2 - L1 - P1) > 0:
            mcmcMu[i] = np.copy(mcmcMuValNext);
            mcmcMuVal = np.copy(mcmcMuValNext);
            if(i >= mcmcNCut) and (i%thinF == 0):  #trim+thin
                iterAcc.append(iteration[i]);
                mcmcMuAcc.append(mcmcMuVal);
        else:
            U = np.random.uniform(0, 1);
            if U < np.exp(L2 + P2 - L1 - P1):
                mcmcMu[i] = np.copy(mcmcMuValNext);
                mcmcMuVal = np.copy(mcmcMuValNext);
                if(i >= mcmcNCut) and (i%thinF == 0):  #trim+thin
                    iterAcc.append(iteration[i]);
                    mcmcMuAcc.append(mcmcMuVal);
            else:
                if i >= mcmcNCut:
                    iterRej.append(iteration[i]);
                    mcmcMuRej.append(mcmcMuValNext);
    rangeR = 2.0/np.sqrt(mcmcN);
    postRangeX = [rangeX[0]*rangeR + np.average(mcmcMuAcc), \
                  rangeX[1]*rangeR + np.average(mcmcMuAcc)];
    postNbins = np.linspace(postRangeX[0], postRangeX[1], binN+1)[:-1];
    posteriorHist = np.zeros(binN);
    for x in mcmcMuAcc:
        if postRangeX[0] < x and x < postRangeX[1]:
            posteriorHist[int(binN*(x-postRangeX[0])\
                                  /(postRangeX[1]-postRangeX[0]))] += 1;


#plots
    fig = plt.figure(figsize=(18, 14));
    gs = gridspec.GridSpec(2, 2);
    ax0 = fig.add_subplot(gs[0]);
    ax1 = fig.add_subplot(gs[1]);
    ax2 = fig.add_subplot(gs[2]);
    ax3 = fig.add_subplot(gs[3]);
    #plot 0
    gaussPlot = gaussian(dataMu, dataSig, nbins);
    ax0.plot(nbins, dataHist, linewidth=2, color="blue", drawstyle="steps-post");
    ax0.plot(nbins, gaussPlot*np.sum(dataHist)/np.sum(gaussPlot), linewidth=2, \
             alpha=0.8, color="red")
    ax0.axhline(y=0, color="black", linestyle="-");
    ax0.axvline(x=np.average(dataPDF), ymin=0, ymax=1, color="green", \
                linestyle="--");
    ax0.set_title("Generated Data", fontsize=24, y=1.03);
    ax0.set_xlabel("x", fontsize=18);
    ax0.set_ylabel("count", fontsize=18);
    ax0.set_xlim(rangeX[0]-1.0, rangeX[1]+1.0);
   
    expVal0 = np.average(dataPDF);
    error0  = np.std(dataPDF)/np.sqrt(dataN);
    digit0  = -int(math.log10(error0)) + 2;
    expVal0r = ("{:." + str(digit0) + "f}").format(expVal0);
    error0r  = ("{:." + str(digit0) + "f}").format(error0);
    freqStr  = "Freq: " + expVal0r + " +/- " + error0r;
    ymin, ymax = ax0.get_ylim();
    font = {"family": "serif", "color": "green", "weight": "bold", "size": 18};
    ax0.text(expVal0, 0.92*(ymax-ymin), freqStr, fontdict=font); 
    #plot 1
    ax1.plot(nbins, priorHist, alpha=1.0, color="blue", drawstyle="steps-post");
    ax1.axhline(y=0, color="black", linestyle="-");
    ymin, ymax = ax1.get_ylim();
    yLow   = (0 - ymin)/(ymax - ymin);
    yHighL = (priorHist[0] - ymin)/(ymax - ymin);
    yHighR = (priorHist[binN-1] - ymin)/(ymax - ymin);
    ax1.axvline(x=nbins[0], ymin=yLow, ymax=yHighL, color="blue", \
                linestyle="-");
    ax1.axvline(x=nbins[binN-1], ymin=yLow, ymax=yHighR, color="blue", \
                linestyle="-");
    ax1.set_title("Pre-defined Prior", fontsize=24, y=1.03);
    ax1.set_xlabel("mu", fontsize=18);
    ax1.set_ylabel("amplitude", fontsize=18);
    ax1.set_xlim(rangeX[0]-1.0, rangeX[1]+1.0);
    #plot 2
    ax2.plot(iteration, mcmcMu, alpha=1.0, color="black",\
             drawstyle="steps-post", zorder=0);
    ax2.scatter(iterAcc, mcmcMuAcc, alpha=0.8, color="blue", marker="o",\
                s=20, zorder=1);
    ax2.scatter(iterRej, mcmcMuRej, alpha=0.8, color="red", marker="x",\
                s=30, zorder=2);
    ax2.axhline(y=np.average(mcmcMuAcc), xmin=0, xmax=1, color="green", \
                linewidth=3, linestyle="--", zorder=3);
    ax2.set_title("MCMC Sampling Trace", fontsize=24, y=1.03);
    ax2.set_xlabel("iteration", fontsize=18);
    ax2.set_ylabel("mu", fontsize=18);
    ax2.set_xlim(-0.03*mcmcN, 1.03*mcmcN);
    #plot 3
    ax3.plot(postNbins, posteriorHist/np.sum(posteriorHist), alpha=1.0, \
             color="blue", drawstyle="steps-post");
    ax3.axhline(y=0, color="black", linestyle="-");
    ax3.axvline(x=np.average(mcmcMuAcc), ymin=0, ymax=1, color="green", \
                linestyle="--");
    ax3.set_title("Estimated Posterior", fontsize=24, y=1.03);
    ax3.set_xlabel("mu", fontsize=18);
    ax3.set_ylabel("amplitude", fontsize=18);
    ax3.set_xlim(postRangeX[0]-1.0*rangeR, postRangeX[1]+1.0*rangeR);
    
    expVal3 = np.average(mcmcMuAcc);
    error3  = np.std(mcmcMuAcc);
    digit3  = -int(math.log10(error3)) + 2;
    expVal3r = ("{:." + str(digit3) + "f}").format(expVal3);
    error3r  = ("{:." + str(digit3) + "f}").format(error3);
    bayeStr  = "Baye: " + expVal3r + " +/- " + error3r;
    ymin, ymax = ax3.get_ylim();
    font = {"family": "serif", "color": "green", "weight": "bold", "size": 18};
    ax3.text(expVal3, 0.92*(ymax-ymin), bayeStr, fontdict=font); 

    if verbosity >= 1:
        print("Freq: " + str(expVal0) + " +/- " + str(error0));
        print("Baye: " + str(expVal3) + " +/- " + str(error3)); 
#save plots
    exepath = os.path.dirname(os.path.abspath(__file__));
    filenameFig = exepath + "/gausMu.png";
    gs.tight_layout(fig);
    plt.savefig(filenameFig);

    if verbosity >= 1:
        print("Creating the following files:");
        print(filenameFig);

if __name__ == "__main__":
    print("\n##############################################################Head");
    main();
    print("##############################################################Tail");




