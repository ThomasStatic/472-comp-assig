from scipy.optimize import linprog 

Xcal: list[str] = ['B', 'G']
Ucal: list[int] = [0, 1]
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
        
def extract_policy(eta: float) -> list[int]:
    c = [0, eta - 1, 0, eta]
    A = [[1,1,1,1], [0.1, 0.9, -0.8, -0.2]]
    b = [1, 0]

    v = linprog(c, A_ub=A, b_ub=b).x
    print(f"Optimal value function for eta={eta}: {v}")

    policy_G: int = 0 if v[0]/(v[0]+v[1]) > v[1]/(v[0]+v[1]) else 1
    policy_B: int = 0 if v[2]/(v[2]+v[3]) > v[3]/(v[2]+v[3]) else 1

    return [policy_G, policy_B]

if __name__ == "__main__":
    for eta in etas:
        policy = extract_policy(eta)
        print(f"Optimal policy for eta={eta}: {policy}")