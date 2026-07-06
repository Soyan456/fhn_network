import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx

# Parameters
a = 0.7
epsilon = 0.08
gamma = 0.8
I = 0.3
k = 0.5

# Network parameters
N = 50        # number of agents
d = 4         # average degree (each node connected to d neighbors)
p = 0.1       # rewiring probability (small world)

# Build Watts-Strogatz network
G = nx.watts_strogatz_graph(N, d, p, seed=42)
adj = list(G.adjacency())

def fhn_network(t, y):
    # y contains [v0, w0, v1, w1, ..., vN-1, wN-1]
    v = y[0::2]  # all v values
    w = y[1::2]  # all w values
    
    dydt = np.zeros(2*N)
    
    for i in range(N):
        neighbors = list(G.neighbors(i))
        coupling = k * sum(v[j] - v[i] for j in neighbors)
        
        dydt[2*i]   = v[i]*(v[i] - a)*(1 - v[i]) - w[i] + I + coupling
        dydt[2*i+1] = epsilon*(v[i] - gamma*w[i])
    
    return dydt

# Initial conditions
# Small fraction of agents start above threshold
np.random.seed(42)
y0 = np.zeros(2*N)
n_initial_snappers = 5  # 5 out of 50 start above threshold

snapper_indices = np.random.choice(N, n_initial_snappers, replace=False)
for i in range(N):
    if i in snapper_indices:
        y0[2*i] = 0.85  # above threshold
    else:
        y0[2*i] = 0.2   # below threshold
    y0[2*i+1] = 0.0

# Solve
t_span = (0, 300)
t_eval = np.linspace(0, 300, 3000)

sol = solve_ivp(fhn_network, t_span, y0, 
                t_eval=t_eval, method='RK45',
                rtol=1e-6, atol=1e-8)

# Extract v values
v_all = sol.y[0::2]  # shape (N, timepoints)

# Plot
plt.figure(figsize=(12, 6))
for i in range(N):
    color = 'tomato' if i in snapper_indices else 'steelblue'
    alpha = 0.8 if i in snapper_indices else 0.3
    plt.plot(sol.t, v_all[i], color=color, alpha=alpha, linewidth=0.8)

plt.axhline(y=a, color='black', linestyle='--', 
            alpha=0.7, label='threshold a')
plt.xlabel('Time')
plt.ylabel('v')
plt.title(f'FHN Network (N={N}, p={p}, k={k}, I={I})')
plt.legend(['threshold', 'initial snappers (red)', 'susceptible (blue)'])
plt.tight_layout()
plt.show()

# Cascade measurement
final_window = v_all[:, int(0.8*len(sol.t)):]
v_means = np.mean(final_window, axis=1)
n_cascaded = np.sum(v_means > 0.5)
print(f"Agents cascaded: {n_cascaded}/{N} ({100*n_cascaded/N:.1f}%)")
print(f"Network clustering: {nx.average_clustering(G):.4f}")
print(f"Network avg path length: {nx.average_shortest_path_length(G):.4f}")
