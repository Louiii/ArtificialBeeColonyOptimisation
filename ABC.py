import numpy as np

def plot(problem, bound, best, name, n=30):
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    (xmin, xmax) = bound
    x = np.outer(np.linspace(xmin, xmax, n), np.ones(n))
    y = x.copy().T
    z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xi, yi = x[i, j], y[i, j]
            z[i, j] = problem(np.array([xi, yi]))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(20, 15)

    [xo, yo], zo = list(best.x), best.f
    ax.scatter(xo, yo, zo, color='red', s=10)

    ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none', alpha=.6)
    plt.title(name)
    plt.savefig(name, dpi=300)
    plt.show()

class Bee:
    def __init__(self, x=[], f=None):
        self.x = x
        self.f = f

def ABC(fn, d, bound, SN, limit, MCN):
    '''  minimise fn over domain x in [bound]^d  '''
    (xmin, xmax) = bound

    # Scout bees initiate food source, random food positions
    X = np.random.uniform(low=xmin, high=xmax, size=(SN, d))
    employed = [Bee(x=X[i], f=fn(X[i])) for i in range(SN)]
    onlooker = employed[:]

    C = np.zeros(SN)
    for it in range(MCN):
        # Employed bees; place on the food sources in the memory
        # -> measure nectar amounts
        for i in range(SN):
            K = list(range(i-1))+list(range(i, SN))
            k = K[np.random.randint(len(K))]

            # j = np.random.randint(d)
            # phi = np.random.uniform(low=-1, high=1)
            # v = employed[i].x
            # v[j] = min(max(v[j] + phi * (v[j] - employed[k].x[j]), xmin), xmax)
            
            phi = np.random.uniform(low=-1, high=1, size=d)
            v = employed[i].x + np.multiply(phi, (employed[i].x - employed[k].x))
            v = np.minimum(np.maximum(v, xmin), xmax)

            fv = fn(v)
            
            if fv < employed[i].f:
                employed[i] = Bee(x=v, f=fv)
            else:
                C[i] += 1
        
        # Onlooker bees; place on the food sources in the memory
        # -> select the food sources
        fit = np.zeros(SN)
        for i in range(SN):
            # fit[i] = -employed[i].f
            if (employed[i].f >= 0):
                fit[i] = 1/(1+employed[i].f)
            else:
                fit[i] = 1+abs(employed[i].f)
        P = fit/sum(fit)
        
        for i in range(SN):
            # K = list(range(i-1))+list(range(i, SN))
            # k = K[np.random.randint(len(K))]
            n = np.random.choice(range(len(P)), p=P/np.sum(P))
            
            K = list(range(n-1))+list(range(n, SN))
            k = K[np.random.randint(len(K))]

            # j = np.random.randint(d)
            # phi = np.random.uniform(low=-1, high=1)
            # v = employed[n].x
            # v[j] = min(max(v[j] + phi * (v[j] - employed[k].x[j]), xmin), xmax)

            phi = np.random.uniform(low=-1, high=1, size=d)
            v = employed[n].x + np.multiply(phi, (employed[n].x - employed[k].x))
            v = np.minimum(np.maximum(v, xmin), xmax)

            fv = fn(v)
            
            if fv < onlooker[n].f:
                onlooker[n] = Bee(x=v, f=fv)
            # else:
            #     C[n] += 1
        
        # Scout bees; send to the search area for discovering new food sources
        # -> determine a scout bee -> send to possible food sources
        # for i in range(SN):
        #     if C[i] >= limit:
        #         employed[i].x = np.random.uniform(low=xmin, high=xmax, size=d)
        #         employed[i].f = fn(employed[i].x)
        #         C[i] = 0
        #         break
        mask = C >= limit
        tot_exh = sum(mask)
        if tot_exh > 0:
            i = np.random.choice(range(SN), p=mask/tot_exh)
            employed[i].x = np.random.uniform(low=xmin, high=xmax, size=d)
            employed[i].f = fn(employed[i].x)
            C[i] = 0

    best = Bee(f=float('inf'))
    for i in range(SN):
        if employed[i].f < best.f: best = employed[i]
        if onlooker[i].f < best.f: best = onlooker[i]
    return best

def greiwank(x):
    return np.sum(np.power(x, 2))/4000 + 1 - np.product(np.cos(np.multiply(x, np.power(np.arange(1,len(x)+1), -0.5))))

def rastrigin(x):
    return np.sum(np.power(x, 2) - 10*np.cos(2*np.pi*x) + 10)

def rosenbrock(x):
    x2 = np.power(x, 2)
    v = np.sum(np.power(x2[:-1] - x[1:], 2)) + (x2[-1]*x[0])**2
    return 100*v + np.sum(np.power(1-x, 2))

def ackley(x):
    return 20+np.e-20*np.exp(-0.2*np.sqrt(np.sum(np.power(x, 2))/len(x)))-np.exp(np.sum(np.cos(2*np.pi*x))/len(x))

def schwefel(x):
    return len(x)*4128.9829-np.sum(x*np.sin(np.sqrt(np.abs(x))))

if __name__=="__main__":
    names = ["Greiwank", "Rastrigin", "Rosenbrock", "Ackley", "Schwefel"]
    problems = [greiwank, rastrigin, rosenbrock, ackley, schwefel]

    d = 2
    for name, problem in zip(names, problems):
        bound = (-4.5, 4.5)
        SN, MCN = 50, 45
        limit = SN*d

        best = ABC(problem, d, bound, SN, limit, MCN)
        plot(problem, bound, best, name, n=100)
