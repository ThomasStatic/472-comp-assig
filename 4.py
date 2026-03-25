import numpy as np
import matplotlib.pyplot as plt

A: np.ndarray = np.array([[1.5, 1, 0, 0], [0, 1.5, 1, 0], [0, 0, 1.5, 1], [0, 0, 0, 1.1]])
C: np.ndarray = np.array([[2, 0, 0, 0]])

def plot_kalman():
    m = np.zeros(4)
    Sigma = np.eye(4)
    x = np.random.normal(size=4)
    
    # y here as in y-axis
    x_y = []
    m_tilde_y = []

    for t in range(150):
        # v is just normal noise
        y = C @ x + np.random.normal()

        m_tilde = m + Sigma @ C.T @ np.linalg.inv(C@Sigma@C.T + 1) @ (y - C @ m)

        x_y.append(x)
        m_tilde_y.append(m_tilde.copy())

        w = np.random.normal(size=4)
        x = A @ x + w
        m = A @ m_tilde
        Sigma = A @ (Sigma - (Sigma @ C.T @ np.linalg.inv(C @ Sigma @ C.T + 1) @ C) @ Sigma) @ A.T + np.eye(4)

    plt.figure()
    plt.title("x_t")
    for i in range(4):
        plt.plot(np.arange(150), np.array(x_y)[:, i], label=f'x[{i}]')
        plt.legend()
    plt.show()

    plt.figure()
    plt.title("m_tilde_t")
    for i in range(4):
        plt.plot(np.arange(150), np.array(m_tilde_y)[:, i], label=f'm_tilde[{i}]')
        plt.legend()
    plt.show()

    plt.figure()
    plt.title("x_t - m_tilde_t")
    for i in range(4):
        plt.plot(np.arange(150), np.array(x_y)[:, i] - np.array(m_tilde_y)[:, i], label=f'x[{i}] - m_tilde[{i}]')
        plt.legend()
    plt.show()

def check_convergence():
    plt.figure()

    initial_sigmas = [np.linalg.matrix_power((idx+1) * np.eye(4), (idx+1)) for idx in range(10)]
    for idx, initial_sigma in enumerate(initial_sigmas):
        Sigma_0 = initial_sigma.copy()
        Sigma_norms = []
        for t in range(150):
            Sigma_1 = A @ (Sigma_0 - Sigma_0 @ C.T @ np.linalg.inv(C @ Sigma_0 @ C.T + 1) @ C @ Sigma_0) @ A.T + np.eye(4)
            # fro = Frobenius norm
            Sigma_norms.append(np.linalg.norm(Sigma_1 - Sigma_0, ord="fro"))
            Sigma_0 = Sigma_1.copy()
        
        plt.plot(np.arange(150), Sigma_norms, label=f'idx={idx}')
    
    plt.legend()
    plt.yscale("log")
    plt.show()

if __name__ == "__main__":
    plot_kalman()
    check_convergence()