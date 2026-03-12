def calc_cost(x, u, eta):
    if x == 'G':
        return u(eta-1)
    return u(eta)

