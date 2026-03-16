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

def policy_iteration(eta) -> tuple[dict[str, float], dict[str, int]]:
    policy_old: dict[str, int] = {'G': 0, 'B': 0}
    policy_new: dict[str, int] = {'G': 0, 'B': 0}
    v: dict[str, float] = {'G': 0, 'B': 0}
    v_old: dict[str, float] = {'G': 0, 'B': 0}
    iteration: int = 0
    while(True):
        iteration += 1

        while (True):
            for x in Xcal:
                v[x] = calc_cost(x, policy_old[x], eta) + beta*compute_future_value(x, policy_old[x], v_old)
            if max(abs(v[x]-v_old[x]) for x in Xcal) < 1e-6:
                break
            v_old = v.copy()
        
        for x in Xcal:
            policy_new[x] = 0 if calc_cost(x, 0, eta) + beta*compute_future_value(x, 0, v) < calc_cost(x, 1, eta) + beta*compute_future_value(x, 1, v) else 1
        if policy_new == policy_old:
            break
        policy_old = policy_new.copy()
    
    print(f"Converged after {iteration} iterations")
    return v, policy_new
        

def compute_future_value(x0: str, u0: int, v: dict[str, float]) -> float:
    next_state_value = 0
    for x1 in Xcal:
        next_state_value += kernel(x0, x1, u0)*v[x1]
    return next_state_value


if __name__ == "__main__":
    for eta in etas:
        print("\nValue Iteration")
        print(f"eta: {eta}")
        v, policy = value_iteration(eta)
        print(f"Value Function: {v}")
        print(f"Policy: {policy}")

        print("\nPolicy Iteration")
        print(f"eta: {eta}")
        v, policy = policy_iteration(eta)
        print(f"Value Function: {v}")
        print(f"Policy: {policy}")