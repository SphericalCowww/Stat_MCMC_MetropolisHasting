# Using MCMC Metropolis Hasting Algorithm to Fit a Gaussian from scratch

Markov chain Monte Carlo (MCMC) Metropolis Hasting is a Bayesian approach to point estimate. In other words, a prior is required for each of the parameters estimated. The MCMC algorithm samples the parameter space according to the prior while updating the prior after each sampling. For each sampling, the Metropolis Hasting algorithm accepts/rejects the update based on an acceptance function based on the priors and the likelihood function of the distribution to fit.

In this example, the parameters are the &mu; and &sigma; sampling from a Gaussian with a sample size of 1,000 and, <br/>
&ensp;&ensp;&mu; = 0.2 and &sigma; = 0.14. <br/>

The code runs on python3 with additional packages:

    pip3 install scipy
    python3 MCMCMH_gausMuSig.py
The code outputs the following image:

<img src="https://github.com/SphericalCowww/Stat_MCMC_MetropolisHasting/blob/main/gausMuSig_Display.png" width="630" height="490">

- Top-left: the blue distribution is the sample drawn from the red Gaussian curve. The red curve is obtained using the frequentist approach, i.e. the point estimate. The estimated $\mu_{freq}$ and $\sigma_{freq}$ are give in purple. The error of $\sigma_{freq}$ is too difficult to evaluate, and so it is left as if it is unavailable. 

- Top-right: the prior distribution for $\mu_{baye}$ and $\sigma_{baye}$, both are Gaussians centered at 0 and have standard deviations of 2.0. 

- Bottom-left: the trace of MCMC for $\sigma_{baye}$ is shown by the black curve. There are 10,000 iteration samples. The blue dots represent the MCMC values accepted by the Metropolis Hasting algorithm and the red crosses rejected. The first 1/10 of the iterations are cut out (trimming) to account for the effects of the initial conditions ($\mu_{baye}(t=0) = 0.0$ and $\sigma_{baye}(t=0)=1.0$); about 2/3 of the accepted values are removed (thinning) to mitigate the effect of autocorrelation from the Markov chain; the gaussian proposal step drawn from the Monte Carlo shrinks in width (annealing) over iterations for faster convergence. 

- Bottom-right: the samples drawn from the MCMC for a distribution for $\mu_{baye}$ and $\sigma_{baye}$, whose values and errors are presented in purple. The Bayesian results are consistent with those of the frequentist. 

The code also outputs the following image:

<img src="https://github.com/SphericalCowww/Stat_MCMC_MetropolisHasting/blob/main/gausMuSigAuto_Display.png" width="630" height="245">

The image shows the autocorrelation of the $\mu$ & $\sigma$ traces up to a step lag of 100. The amplitude mean is subtracted and normalized to the std to the corresponding $\mu$ & $\sigma$ values. The thinning step is chosen to be 3, i.e.~ randomly drop 2/3 of the accepted proposals; from their autocorrelation, we can see this thinning step is too small.

Note: the cross-correlation definition used to compute the autocorrelation is "of deterministic signals", which is different from the cross-correlation "of random vectors" (<a href="https://en.wikipedia.org/wiki/Cross-correlation">Wiki</a>).

References:
- StataCorp LLC's Youtube channel (2016) (<a href="https://www.youtube.com/watch?v=OTO1DygELpY">Youtube</a>)
