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
from scipy import special

import warnings
warnings.filterwarnings("ignore");


def uniform(lowX, highX, x):
    X = np.array(x);
    X[(X < lowX)|(highX < X)] = lowX - 1.0;
    X[(lowX <= X)&(X <= highX)] = 1.0/(highX-lowX);
    X[X < lowX] = 0;
    return X;
def gaussian(mu, sig, x):
    X = np.array(x);
    val = np.exp(-np.power(X-mu,2.0)/(2.0*np.power(sig,2.0)))\
         *(1.0/(sig*np.sqrt(2.0*np.pi)));
    return val;
def logFunc(x):
    X = np.array(x);
    X[x < np.finfo(float).eps] = np.exp(-np.Inf);
    return np.log(x);

def priorUni(a, b):
    return lambda x : uniform(a, b, x);
def priorGaus(a, b):
    return lambda x : gaussian(a, b, x);
def likelihoodGaus(x):
    return lambda a, b: gaussian(a, b, x);
    

#data:  gaussian
#prior: uniform
#mcmc:  gaussian
def main():
    verbosity = 1;
    binN = 200+1;
    rangeX = [-10.0, 10.0];

    np.random.seed(1);
    dataMu = 0.2;
    dataSig = 1.4;
    dataN = 1000;

    priorMu  = priorGaus(0.0, 2.0);
    priorSig = priorGaus(0.0, 2.0);

    mcmcMuMuInit  = 0.0;
    mcmcSigMuInit = 1.0;
    mcmcMuSig  = 0.1;
    mcmcSigSig = 0.1;
    mcmcN = 10000;
    mcmcNCut = int(mcmcN/10);   #trimming
    thinF = 3;                  #thinning
    freezF = 0.1;               #annealing

    traceOption = "sig";        #for display
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
    likelihood = likelihoodGaus(dataPDF);
    if priorMu(mcmcMuMuInit) == 0:
        print("Run stops.");
        print("The initial mu value is incompatible with the prior.");
        exit(0);
    if priorSig(mcmcSigMuInit) == 0:
        print("Run stops.");
        print("The initial sigma value is incompatible with the prior.");
        exit(0);

    iteration = np.linspace(0, mcmcN, mcmcN+1)[:-1];
    mcmcMuVal  = np.copy(mcmcMuMuInit);
    mcmcSigVal = np.copy(mcmcSigMuInit);
    mcmcMu  = np.zeros(mcmcN);
    mcmcSig = np.zeros(mcmcN);
    iterAcc    = [];
    mcmcMuAcc  = [];
    mcmcSigAcc = [];
    iterRej    = [];
    mcmcMuRej  = [];
    mcmcSigRej = [];
    for i in range(0, mcmcN):
        #MCMC
        mcmcMuSigVal   = mcmcMuSig/(1.0 + freezF*np.sqrt(i));       #anneal
        mcmcSigSigVal  = mcmcSigSig/(1.0 + freezF*np.sqrt(i));      #anneal
        mcmcMuValNext  = np.random.normal(mcmcMuVal,  mcmcMuSigVal);
        mcmcSigValNext = np.random.normal(mcmcSigVal, mcmcSigSigVal);
        #Bayesian likelihood/prior
        likelihood1 = likelihood(mcmcMuVal, mcmcSigVal);
        prior1      = priorMu(mcmcMuVal)*priorSig(mcmcSigVal);
        likelihood2 = likelihood(mcmcMuValNext, mcmcSigValNext);
        prior2      = priorMu(mcmcMuValNext)*priorSig(mcmcSigValNext);
        L1 = np.sum(logFunc(likelihood1));
        P1 = logFunc(prior1);
        L2 = np.sum(logFunc(likelihood2));
        P2 = logFunc(prior2);
        #Metropolis Hasting
        acceptance = L2 + P2 - L1 - P1;
        mcmcMu[i] = np.copy(mcmcMuVal);
        mcmcSig[i] = np.copy(mcmcSigVal);
        if acceptance > 0:
            mcmcMu[i] = np.copy(mcmcMuValNext);
            mcmcMuVal = np.copy(mcmcMuValNext);
            mcmcSig[i] = np.copy(mcmcSigValNext);
            mcmcSigVal = np.copy(mcmcSigValNext);
            if(i >= mcmcNCut) and (i%thinF == 0):  #trim+thin
                iterAcc.append(iteration[i]);
                mcmcMuAcc.append(mcmcMuValNext);
                mcmcSigAcc.append(mcmcSigValNext);
        else:
            U = np.random.uniform(0, 1);
            if U < np.exp(acceptance):
                mcmcMu[i] = np.copy(mcmcMuValNext);
                mcmcMuVal = np.copy(mcmcMuValNext);
                mcmcSig[i] = np.copy(mcmcSigValNext);
                mcmcSigVal = np.copy(mcmcSigValNext);
                if(i >= mcmcNCut) and (i%thinF == 0):  #trim+thin
                    iterAcc.append(iteration[i]);
                    mcmcMuAcc.append(mcmcMuValNext);
                    mcmcSigAcc.append(mcmcSigValNext);
            else:
                if i >= mcmcNCut:
                    iterRej.append(iteration[i]);
                    mcmcMuRej.append(mcmcMuValNext);
                    mcmcSigRej.append(mcmcSigValNext);
    #converting list to numpy arrays
    iterAcc    = np.array(iterAcc);
    mcmcMuAcc  = np.array(mcmcMuAcc);
    mcmcSigAcc = np.array(mcmcSigAcc);
    iterRej    = np.array(iterRej);
    mcmcMuRej  = np.array(mcmcMuRej);
    mcmcSigRej = np.array(mcmcSigRej);
    #prepend the initial values
    iteration = np.insert(iteration, 0, -1);
    mcmcMu    = np.insert(mcmcMu, 0, mcmcMuMuInit);
    mcmcSig   = np.insert(mcmcSig, 0, mcmcSigMuInit);
    iterAcc    = np.insert(iterAcc, 0, -1);
    mcmcMuAcc  = np.insert(mcmcMuAcc, 0, mcmcMuMuInit);
    mcmcSigAcc = np.insert(mcmcSigAcc, 0, mcmcSigMuInit);



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
    ax0.axvline(x=np.average(dataPDF), ymin=0, ymax=1, color="darkviolet", \
                linestyle="--");
    ax0.set_title("Generated Data", fontsize=24, y=1.03);
    ax0.set_xlabel("x", fontsize=18);
    ax0.set_ylabel("count", fontsize=18);
    ax0.set_xlim(rangeX[0]-1.0, rangeX[1]+1.0);
   
    expMu0 = np.average(dataPDF);
    errMu0 = np.std(dataPDF)/np.sqrt(dataN);
    digit0 = 3;
    expMu0r = "";
    errMu0r = ""; 
    if errMu0 != 0:
        digit0 = -int(math.log10(errMu0)) + 2;
        expMu0r = ("{:." + str(digit0) + "f}").format(expMu0);
        errMu0r = ("{:." + str(digit0) + "f}").format(errMu0);
    else:
        expMu0r = ("{:.3f}").format(expMu0);
        errMu0  = "NaN";
        errMu0r = "NaN";
    expSig0  = np.sqrt(np.var(dataPDF));
    #actuarialdatasience.com/standard_deviation_standard_error.html
    #but the gamma functions are very difficult to deal with
    errSig0 = 0;
    digit0 = 3;
    expSig0r = "";
    errSig0r = "";
    if errSig0 != 0:
        digit0 = -int(math.log10(errSig0)) + 2;
        expSig0r = ("{:." + str(digit0) + "f}").format(expSig0);
        errSig0r = ("{:." + str(digit0) + "f}").format(errSig0);
    else:
        expSig0r = ("{:.3f}").format(expSig0);
        errSig0  = "NaN";
        errSig0r = "NaN"; 
    freqStr = " Freq: \n";
    freqStr = freqStr + "    " + expMu0r + "\u00B1" + errMu0r + " (mu)\n";
    freqStr = freqStr + "    " + expSig0r + "\u00B1" + errSig0r + " (sig)";
    ymin, ymax = ax0.get_ylim();
    font={"family": "serif", "color": "darkviolet", "weight": "bold", "size": 18};
    ax0.text(expMu0, 0.822*(ymax-ymin), freqStr, fontdict=font, zorder=10);
    #plot 1
    mubins  = np.linspace(rangeX[0], rangeX[1], binN);
    sigbins = np.linspace(rangeX[0], rangeX[1], binN);
    X, Y = np.meshgrid(mubins, sigbins);
    Z = priorMu(X)*priorSig(Y);
    res1 = ax1.imshow(Z, interpolation="nearest", cmap="jet");
    fig.colorbar(res1, ax=ax1);
    plt.sca(ax1);
    ax1.set_title("Pre-defined Prior", fontsize=24, y=1.03);
    ax1.set_xlabel("mu", fontsize=18);
    ax1.set_ylabel("sigma", fontsize=18);
    ax1.set_xlim(-np.ceil(binN*0.05), np.floor(binN*1.05));
    ax1.set_ylim(-np.ceil(binN*0.05), np.floor(binN*1.05));

    PSFStep = int(math.ceil((binN-1)/10.0));
    nbinsTicks = nbins[0::PSFStep];
    nbinsRange = np.array(range(len(nbinsTicks)))*PSFStep;
    plt.xticks(nbinsRange, nbinsTicks);
    plt.yticks(nbinsRange, nbinsTicks);
    #plot 2
    tracePlot  = mcmcMu;
    accPlot    = mcmcMuAcc;
    rejPlot    = mcmcMuRej;
    traceTitle = "mu";
    if "sig" in traceOption:
        tracePlot  = mcmcSig;
        accPlot    = mcmcSigAcc;
        rejPlot    = mcmcSigRej;
        traceTitle = "sigma";
    ax2.plot(iteration, tracePlot, alpha=1.0, color="black",\
             drawstyle="steps-post", zorder=0);
    ax2.scatter(iterAcc, accPlot, alpha=0.8, color="blue", marker="o",\
                s=20, zorder=2);
    ax2.scatter(iterRej, rejPlot, alpha=0.8, color="red", marker="x",\
                s=30, zorder=1);
    ax2.axhline(y=np.average(accPlot), xmin=0, xmax=1, color="darkviolet", \
                linewidth=3, linestyle="--", zorder=3);
    ax2.set_title("MCMC Sampling Trace", fontsize=24, y=1.03);
    ax2.set_xlabel("iteration", fontsize=18);
    ax2.set_ylabel(traceTitle, fontsize=18);
    ax2.set_xlim(-0.03*mcmcN, 1.03*mcmcN);
    #plot 3
    res3 = ax3.hexbin(mcmcMuAcc, mcmcSigAcc, cmap="jet",\
                      gridsize=max(10, int(np.sqrt(mcmcN)/2.0)));
    fig.colorbar(res3, ax=ax3).ax.zorder = -1;
    plt.sca(ax3);
    ax3.axvline(x=np.average(mcmcMuAcc), ymin=0, ymax=1, color="darkviolet", \
                linewidth=3, linestyle="--");
    ax3.axhline(y=np.average(mcmcSigAcc), xmin=0, xmax=1, color="darkviolet", \
                linewidth=3, linestyle="--");
    ax3.set_title("Estimated Posterior", fontsize=24, y=1.03);
    ax3.set_xlabel("mu", fontsize=18);
    ax3.set_ylabel("sigma", fontsize=18);
    expMu3 = np.average(mcmcMuAcc);
    errMu3 = np.std(mcmcMuAcc);
    digit3 = 3;
    expMu3r = "";
    errMu3r = "";
    if errMu3 != 0:
        digit3 = -int(math.log10(errMu3)) + 2;
        expMu3r = ("{:." + str(digit3) + "f}").format(expMu3);
        errMu3r = ("{:." + str(digit3) + "f}").format(errMu3);
    else:
        expMu3r = ("{:.3f}").format(expMu3);
        errMu3  = "NaN";
        errMu3r = "NaN";
    expSig3 = np.average(mcmcSigAcc);
    errSig3 = np.std(mcmcSigAcc);
    digit3  = 3;
    expSig3r = "";
    errSig3r = "";
    if errSig3 != 0:
        digit3 = -int(math.log10(errSig3)) + 2;
        expSig3r = ("{:." + str(digit3) + "f}").format(expSig3);
        errSig3r = ("{:." + str(digit3) + "f}").format(errSig3);
    else:
        expSig3r = ("{:.3f}").format(expSig3);
        errSig3  = "NaN";
        errSig3r = "NaN";
    bayeStr = " Baye: \n";
    bayeStr = bayeStr + "    " + expMu3r  + "\u00B1" + errMu3r + " (mu)\n";
    bayeStr = bayeStr + "    " + expSig3r + "\u00B1" + errSig3r + " (sig)";
    ymin, ymax = ax3.get_ylim();
    font={"family": "serif", "color": "darkviolet", "weight": "bold", "size": 18};
    ax3.text(expMu3, ymin+0.867*(ymax-ymin), bayeStr, fontdict=font, zorder=10);

    if verbosity >= 1:
        print("Freq: ");
        print(str(expMu0)  + " \u00B1 " + str(errMu0)  + " (mu)");
        print(str(expSig0) + " \u00B1 " + str(errSig0) + " (sigma)");
        print("Baye: ");
        print(str(expMu3)  + " \u00B1 " + str(errMu3)  + " (mu)");
        print(str(expSig3) + " \u00B1 " + str(errSig3) + " (sigma)");
#save plots
    exepath = os.path.dirname(os.path.abspath(__file__));
    filenameFig = exepath + "/gausMuSig.png";
    gs.tight_layout(fig);
    plt.savefig(filenameFig);

    if verbosity >= 1:
        print("Creating the following files:");
        print(filenameFig);

if __name__ == "__main__":
    print("\n##############################################################Head");
    main();
    print("##############################################################Tail");




