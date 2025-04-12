import numpy as np
from scipy.stats import norm


# Ex. 1: find P(Z<2), where Z=N(0,1)
# we can evaluate the pdf of the Normal Dist. but dont have the cfd
# approximate the integral [P(Z<2)] by 1/N sum (f(x)h(x)), 
# where f(x) is the cdf and h(x) is an indicator fct and 1 if x < 2
num_samples = 10000
sample = np.random.normal(0, 1, num_samples)
est = np.sum(sample<2) / len(sample)
theoretical_quantile = norm.cdf(x=2, loc=0, scale=1)

# error estimate via bootstrap
B = 10000
est_boot = [np.sum(np.random.normal(0, 1, num_samples)<2)/num_samples for _ in range(B)]
sd_boot = (np.mean( [(x - np.mean(est_boot))**2 for x in est_boot] ))**(1/2)
###############################




print("done")
