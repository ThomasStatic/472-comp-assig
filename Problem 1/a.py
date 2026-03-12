Xcal: list[str] = ['B', 'G']
Ucal: list[int] = [0, 1]
beta: float = 0.5
M: int = 1
etas: list[float] = [0.99, 0.7, 0.01]

def calc_cost(x: str, u: int, eta: float) -> float:
    return u*(eta-M) if x == 'G' else u*(eta)

def kernel(x0: str, x1: str, u: int) -> float:
    if x1 == 'G':
        if x0 == 'G':
            return 0.1 if u == 1 else 0.9
        else:
            return 0.2 if u == 1 else 0.8
    else:
        if x0 == 'G':
            return 0.9 if u == 1 else 0.1
        else:
            return 0.8 if u == 1 else 0.2


def value_iteration(eta) -> tuple[dict[str, float], dict[str, int]]:
    v_0: dict[str, float] = {'G': 0, 'B': 0}
    v_1: dict[str, float] = {'G': 0, 'B' : 0}
    policy: dict[str, int] = {'G': 0, 'B': 0}
    iteration: int = 0
    while(True):
        iteration += 1
        for x in Xcal:
            # we need the minimum of the two actions, so I can just default set it to the first one
            v_1[x] = calc_cost(x, 0, eta) + beta*(kernel(x, 'G', 0)*v_0['G'] + kernel(x, 'B', 0)*v_0['B'])
            
            
            better_cost = min(v_1[x], calc_cost(x, 1, eta) + beta*(kernel(x, 'G', 1)*v_0['G'] + kernel(x, 'B', 1)*v_0['B']))
            policy[x] = 0 if better_cost == v_1[x] else 1
            v_1[x] = better_cost
            
        if max(abs(v_1[x]-v_0[x]) for x in Xcal) < 1e-6:
            print(f"Converged after {iteration} iterations")
            return v_1, policy
        v_0 = v_1.copy()

if __name__ == "__main__":
    print("Value Iteration")
    for eta in etas:
        print(f"eta: {eta}")
        v1, policy = value_iteration(eta)
        print(f"Value Function: {v1}")
        print(f"Policy: {policy}")