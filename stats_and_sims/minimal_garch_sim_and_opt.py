import numpy as np, scipy.optimize as opt

def sim_garch(T, ω, α, β):
    y, h = np.empty(T), np.empty(T)
    h[0] = ω/(1-α-β)
    y[0] = np.sqrt(h[0])*np.random.randn()
    for t in range(1, T):
        h[t] = ω + α*y[t-1]**2 + β*h[t-1]
        y[t] = np.sqrt(h[t])*np.random.randn()
    return y

def nll(params, y):
    ω, α, β = params
    if ω<=0 or α<0 or β<0 or α+β>=1:        # constraints
        return 1e10
    h = np.empty_like(y)
    h[0] = y.var()
    ll = 0.0
    for t in range(1, len(y)):
        h[t] = ω + α*y[t-1]**2 + β*h[t-1]
        ll += 0.5*(np.log(2*np.pi) + np.log(h[t]) + y[t]**2/h[t])
    return ll

y = sim_garch(1000, 0.1, 0.05, 0.9)
result = opt.minimize(nll, x0=[0.2, 0.1, 0.8], args=(y,), method='Nelder-Mead')
print("ω̂, α̂, β̂ =", result.x)
