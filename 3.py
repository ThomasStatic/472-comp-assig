import random
from typing import Literal

Xcal: list[Literal[0, 1]] = [0, 1]
Ucal: list[Literal[0, 1]] = [0, 1]
beta: float = 0.5
N_values: list[int] = [0, 1, 2, 3]
eta1 = 0.4
eta2 = 0.2
alpha = 0.6
C = 2
R= 1

def state_kernel(x0: Literal[0, 1], x1: Literal[0, 1], u: Literal[0, 1]) -> float:
    if x1 == 1:
        if x0 == 1:
            return 1-eta1 if u == 0 else 1-eta2
        else:
            return 0 if u == 0 else alpha
    else:
        if x0 == 1:
            return eta1 if u == 0 else eta2
        else:
            return 1 if u == 0 else 1-alpha

def sample_observation(x: Literal[0, 1]) -> Literal[0, 1]:
    if x == 1:
        return 1 if random.random() < 0.9 else 0
    else:
        return 0 if random.random() < 0.9 else 1
    
def calc_cost(x: Literal[0, 1], u: Literal[0, 1]) -> float:
    if u == 1:
        return R
    if x == 0:
        return 0
    return C

def q_learning(N: int):
    X0: Literal[0, 1] = 0
    Xt: Literal[0, 1] = X0
    Y0: Literal[0, 1] = 0
    Yt: Literal[0, 1] = Y0

    Q = {}
    visits = {}

    obs = [-1]*N + [Y0]
    acts = [-1]*N
    It = tuple(obs + acts)

    for step in range(10000):
        epsilon: float = 100/(100 + step)

        if random.random() < epsilon:
            u: Literal[0, 1] = random.choice(Ucal)
        else:
            min_Q = min(Q.get((It, u), 0) for u in Ucal)
            u: Literal[0, 1] = random.choice([u for u in Ucal if Q.get((It, u), 0) == min_Q])
        
        next_Xt: Literal[0, 1] = 1 if random.random() < state_kernel(Xt, 1, u) else 0
        next_Yt: Literal[0, 1] = sample_observation(next_Xt)
        cost_t = calc_cost(Xt, u)

        obs = obs[1:] + [next_Yt]
        acts = acts[1:] + [u]
        next_It = tuple(obs + acts)

        visits[(It, u)] = visits.get((It, u), 0) + 1
        learning_rate: float = 1/(1 + visits[(It, u)])

        Q[(It, u)] = (1-alpha)*Q.get((It, u), 0) + learning_rate*(cost_t + beta*min(Q.get((next_It, u), 0) for u in Ucal))

        Xt, Yt, It = next_Xt, next_Yt, next_It

    
    
    return Q

if __name__ == "__main__":
    for N in N_values:
        print(f"Q-learning with N={N}")
        Q = q_learning(N)
        print(f"Learned Q-values: {Q}")

