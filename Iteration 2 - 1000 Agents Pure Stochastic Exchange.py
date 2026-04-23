import numpy as np
import matplotlib.pyplot as plt


def gini(array):
    """Calculate the Wealth Gini coefficient of a numpy array."""
    array = np.sort(array)
    if np.sum(array) == 0: return 1.0
    n = array.shape[0]
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


# --- Simulation Parameters ---
num_agents = 1000
starting_wealth = 1000.0
bet_percentage = 0.10
growth_percentage = 0.10
floor_percentage = 0.10
num_rounds = 1000
trials = 200  # Number of Monte Carlo trials to average the Origin Stories

# Trackers for Origin Stories
top_1_stats = {'g1': [], 'g2': [], 'g3': [], 'w23': [], 'floor': []}
top_10_stats = {'g1': [], 'g2': [], 'g3': [], 'w23': [], 'floor': []}

print(f"Running {trials} trials of {num_agents} agents for {num_rounds} rounds...")

for t in range(trials):
    wealth = np.full(num_agents, starting_wealth)
    g1_win = np.zeros(num_agents, dtype=bool)
    g2_win = np.zeros(num_agents, dtype=bool)
    g3_win = np.zeros(num_agents, dtype=bool)
    early_wins = np.zeros(num_agents, dtype=int)
    hit_floor = np.zeros(num_agents, dtype=bool)

    # Only track graph data for the first trial to save memory
    if t == 0:
        gini_history = [gini(wealth)]
        wealth_snapshots = {0: wealth.copy()}
        snapshot_rounds = [10, 50, 200, num_rounds]

    for round_num in range(1, num_rounds + 1):
        # --- 1. Bets ---
        indices = np.arange(num_agents)
        np.random.shuffle(indices)
        idx_a = indices[0::2]
        idx_b = indices[1::2]

        w_a = wealth[idx_a]
        w_b = wealth[idx_b]

        total_pair = w_a + w_b
        intended_bet = total_pair * bet_percentage
        actual_bet = np.minimum(intended_bet, np.minimum(w_a, w_b))

        a_wins = np.random.rand(len(idx_a)) < 0.5
        b_wins = ~a_wins

        wealth[idx_a[a_wins]] += actual_bet[a_wins]
        wealth[idx_b[a_wins]] -= actual_bet[a_wins]
        wealth[idx_b[b_wins]] += actual_bet[b_wins]
        wealth[idx_a[b_wins]] -= actual_bet[b_wins]

        round_winners = np.zeros(num_agents, dtype=bool)
        round_winners[idx_a[a_wins]] = True
        round_winners[idx_b[b_wins]] = True

        if round_num == 1:
            g1_win = round_winners.copy()
            early_wins += round_winners
        elif round_num == 2:
            g2_win = g1_win & round_winners
            early_wins += round_winners
        elif round_num == 3:
            g3_win = g2_win & round_winners
            early_wins += round_winners

        # --- 2. Growth and Dynamic Floor ---
        total_wealth = np.sum(wealth)
        system_growth = total_wealth * growth_percentage
        avg_wealth = total_wealth / num_agents

        # The floor is dynamic: pegged to the average wealth of the growing system
        current_floor_threshold = avg_wealth * floor_percentage
        hit_floor = hit_floor | (wealth < current_floor_threshold)

        growth_allocation = system_growth * (wealth / total_wealth)
        min_growth = (system_growth / num_agents) * floor_percentage

        floor_mask = growth_allocation < min_growth
        if np.any(floor_mask):
            shortfall = np.sum(min_growth - growth_allocation[floor_mask])
            growth_allocation[floor_mask] = min_growth
            rich_mask = ~floor_mask
            rich_wealth = wealth[rich_mask]
            rich_total = np.sum(rich_wealth)
            if rich_total > 0:
                growth_allocation[rich_mask] -= shortfall * (rich_wealth / rich_total)
                growth_allocation[rich_mask] = np.maximum(0, growth_allocation[rich_mask])

        wealth += growth_allocation

        if t == 0:
            gini_history.append(gini(wealth))
            if round_num in snapshot_rounds:
                wealth_snapshots[round_num] = wealth.copy()

    # --- End of Trial Processing ---
    sorted_indices = np.argsort(wealth)[::-1]
    top_1_pct_idx = sorted_indices[:int(num_agents * 0.01)]
    top_10_pct_idx = sorted_indices[:int(num_agents * 0.10)]


    def append_stats(indices, stats_dict):
        n = len(indices)
        stats_dict['g1'].append(np.sum(g1_win[indices]) / n * 100)
        stats_dict['g2'].append(np.sum(g2_win[indices]) / n * 100)
        stats_dict['g3'].append(np.sum(g3_win[indices]) / n * 100)
        stats_dict['w23'].append(np.sum(early_wins[indices] >= 2) / n * 100)
        stats_dict['floor'].append(np.sum(hit_floor[indices]) / n * 100)


    append_stats(top_1_pct_idx, top_1_stats)
    append_stats(top_10_pct_idx, top_10_stats)

print(f"\n--- Top 1% Origin Stories ---")
for k in top_1_stats: print(f"{k}: {np.mean(top_1_stats[k]):.1f}%")
print(f"\n--- Top 10% Origin Stories ---")
for k in top_10_stats: print(f"{k}: {np.mean(top_10_stats[k]):.1f}%")

# --- Graphs ---
plt.figure(figsize=(10, 6))
plt.plot(gini_history, color='purple', linewidth=3)
plt.title('Gini Coefficient Over Time (N=1,000 | 10% Bet, 10% Growth, 10% Floor)')
plt.xlabel('Number of Rounds')
plt.ylabel('Wealth Gini Coefficient')
plt.grid(True, alpha=0.3)
plt.axhline(0.85, color='gray', linestyle='--', label='Highly Unequal Real-World Economy (~0.85)')
plt.legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Wealth Distribution Evolution (N=1,000 | Growth + Floor Included)', fontsize=16, fontweight='bold')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, (rnd, ax) in enumerate(zip(snapshot_rounds, axs.flatten())):
    data = wealth_snapshots[rnd]
    ax.hist(data, bins=40, color=colors[idx], edgecolor='black', alpha=0.7)
    ax.set_title(f'Round {rnd}')
    ax.set_xlabel('Wealth')
    ax.set_ylabel('Number of Agents (Log Scale)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()