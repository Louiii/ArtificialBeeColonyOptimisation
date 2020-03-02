import numpy as np

def plot(problem, bound, best, name, n=30):
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    (xmin, xmax, _) = bound
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

    [xo, yo], zo = list(best.loc), best.cost
    ax.scatter(xo, yo, zo, color='red', s=10)

    ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none', alpha=.6)
    plt.title(name)
    plt.savefig(name, dpi=300)
    plt.show()

class Bee:
    def __init__(self, loc=[], cost=None):
        self.loc = loc
        self.cost = cost

def ABC(fn, d, bound, param):
    '''  minimise fn over domain x in [bound]^d  '''
    (n_iter, n) = param
    (xmin, xmax, a) = bound

    lim = round(d*n)
    
    best = Bee(cost=float('inf'))
    bee = [Bee() for _ in range(n)]

    # Scout bees initiate food source
    # -> randomly select a set of food source positions for the bees
    for i in range(n):
        bee[i].loc = np.random.uniform(low=xmin, high=xmax, size=d)
        bee[i].cost = fn(bee[i].loc)
        if bee[i].cost < best.cost:
            best = bee[i]

    C = np.zeros(n)

    for it in range(n_iter):
        # Employed bees; place on the food sources in the memory
        # -> measure nectar amounts
        for i in range(n):
            K = list(range(i-1))+list(range(i, n))
            k = K[np.random.randint(len(K))]
            
            phi = a*np.random.uniform(-1,1,d)[0]
            
            newbee = Bee()
            newbee.loc = bee[i].loc + np.multiply(phi, (bee[i].loc - bee[k].loc))
            newbee.loc = np.minimum(np.maximum(newbee.loc, xmin), xmax)
            newbee.cost = fn(newbee.loc)
            
            if newbee.cost < bee[i].cost:
                bee[i] = newbee
            else:
                C[i] += 1
        
        # Onlooker bees; place on the food sources in the memory
        # -> select the food sources
        fit = np.zeros(n)
        for i in range(n):
            if (bee[i].cost >= 0):
                fit[i] = 1/(1+bee[i].cost)
            else:
                fit[i] = 1+abs(bee[i].cost)
        P = fit/sum(fit)
        
        for i in range(n):
            j = np.random.choice(range(len(P)), p=P)
            K = list(range(j-1))+list(range(j, n))
            k = K[np.random.randint(len(K))]
            
            phi = a*np.random.uniform(-1,1,d)
            
            newbee = Bee()
            newbee.loc = bee[j].loc + np.multiply(phi, (bee[j].loc - bee[k].loc))
            newbee.loc = np.minimum(np.maximum(newbee.loc, xmin), xmax)
            newbee.cost = fn(newbee.loc)
            
            if newbee.cost < bee[i].cost:
                bee[i] = newbee
            else:
                C[i] += 1
        
        # Scout bees; send to the search area for discovering new food sources
        # -> determine the scout bees -> send to possible food sources
        for i in range(n):
            if C[i] >= lim:
                bee[i].loc = np.random.uniform(low=xmin, high=xmax, size=d)
                bee[i].cost = fn(bee[i].loc)
                C[i] = 0
        
        for i in range(n):
            if bee[i].cost < best.cost:
                best = bee[i]
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
        bound = (-4.5, 4.5, 1)
        param = (50, 45)

        best = ABC(problem, d, bound, param)
        plot(problem, bound, best, name, n=100)
