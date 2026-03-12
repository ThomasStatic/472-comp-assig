Xcal = {'B', 'G'}
Ucal = {0, 1}
beta = 0.5
M = 1

def calc_cost(x, u, eta):
    return u(eta-M) if x == 'G' else u(eta)
