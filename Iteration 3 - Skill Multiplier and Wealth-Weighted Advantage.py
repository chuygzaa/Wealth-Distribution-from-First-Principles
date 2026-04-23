import numpy as np
import matplotlib.pyplot as plt

def gini(array):
    array = np.sort(array)
    total = np.sum(array)
    if total == 0: return 1.0
    array = array / total
    n = array.shape[0]
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * array)) / n


num_agents = 1000
starting_wealth = 1000.0
#More realistic relative to earlier iterations
bet_percentage = 0.05  # Down to 5% margins
growth_percentage = 0.03  # Down to 3% GDP growth
floor_percentage = 0.10
num_rounds = 1000
trials = 20

top_1_stats = {'g1': [], 'g2': [], 'g3': [], 'w23': [], 'floor': [], 'skill': []}
top_10_stats = {'g1': [], 'g2': [], 'g3': [], 'w23': [], 'floor': [], 'skill': []}
bottom_50_skill = []

print(f"Running {trials} trials. 3% Growth, 5% Bets, and the SKILL MULTIPLIER...")

for t in range(trials):
    wealth = np.full(num_agents, starting_wealth)

    # Mean = 1.0, Standard Deviation = 0.2
    # Clip between 0.2 (terrible) and 2.0 (absolute genius)
    skills = np.random.normal(1.0, 0.2, num_agents)
    skills = np.clip(skills, 0.2, 2.0)

    g1_win = np.zeros(num_agents, dtype=bool)
    g2_win = np.zeros(num_agents, dtype=bool)
    g3_win = np.zeros(num_agents, dtype=bool)
    early_wins = np.zeros(num_agents, dtype=int)
    hit_floor = np.zeros(num_agents, dtype=bool)

    if t == 0:
        gini_history = [gini(wealth)]
        wealth_snapshots = {0: wealth.copy()}
        snapshot_rounds = [10, 50, 200, num_rounds]

    for round_num in range(1, num_rounds + 1):
        indices = np.arange(num_agents)
        np.random.shuffle(indices)

        idx_a = indices[0::2]
        idx_b = indices[1::2]

        w_a = wealth[idx_a]
        w_b = wealth[idx_b]

        s_a = skills[idx_a]
        s_b = skills[idx_b]

        # Bet Calculation
        pair_total = w_a + w_b
        intended_bet = pair_total * bet_percentage
        actual_bet = np.minimum(intended_bet, np.minimum(w_a, w_b))

        advantage_a = w_a * s_a
        advantage_b = w_b * s_b
        safe_total_adv = np.where((advantage_a + advantage_b) == 0, 1e-9, advantage_a + advantage_b)

        prob_a_wins = advantage_a / safe_total_adv

        a_wins = np.random.rand(len(idx_a)) < prob_a_wins
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

        # Global Growth & UBI Floor (Pegged to Average Slice)
        total_wealth = np.sum(wealth)
        system_growth = total_wealth * growth_percentage

        avg_growth_slice = system_growth / num_agents
        min_growth = avg_growth_slice * floor_percentage

        growth_allocation = system_growth * (wealth / total_wealth)

        poor_mask = growth_allocation < min_growth
        hit_floor = hit_floor | poor_mask

        shortfall = np.sum(min_growth - growth_allocation[poor_mask])
        growth_allocation[poor_mask] = min_growth

        rich_mask = ~poor_mask
        W_rich = np.sum(wealth[rich_mask])

        if W_rich > 0:
            growth_allocation[rich_mask] -= shortfall * (wealth[rich_mask] / W_rich)
            growth_allocation[rich_mask] = np.maximum(min_growth, growth_allocation[rich_mask])

        wealth += growth_allocation

        # Prevent Overflow
        if np.max(wealth) > 1e150:
            wealth /= 1e100

        if t == 0:
            gini_history.append(gini(wealth))
            if round_num in snapshot_rounds:
                wealth_snapshots[round_num] = wealth.copy()

    sorted_indices = np.argsort(wealth)[::-1]
    top_1_pct_idx = sorted_indices[:int(num_agents * 0.01)]
    top_10_pct_idx = sorted_indices[:int(num_agents * 0.10)]
    bottom_50_pct_idx = sorted_indices[int(num_agents * 0.50):]

    def append_stats(indices, stats_dict):
        n = len(indices)
        stats_dict['g1'].append(np.sum(g1_win[indices]) / n * 100)
        stats_dict['g2'].append(np.sum(g2_win[indices]) / n * 100)
        stats_dict['g3'].append(np.sum(g3_win[indices]) / n * 100)
        stats_dict['w23'].append(np.sum(early_wins[indices] >= 2) / n * 100)
        stats_dict['floor'].append(np.sum(hit_floor[indices]) / n * 100)
        stats_dict['skill'].append(np.mean(skills[indices]))


    append_stats(top_1_pct_idx, top_1_stats)
    append_stats(top_10_pct_idx, top_10_stats)
    bottom_50_skill.append(np.mean(skills[bottom_50_pct_idx]))

print(f"\n--- Origin Stories (Averaged over {trials} trials) ---")
print(f"System Average Skill: 1.00")
print(f"Top 1% stats:")
for k in ['g1', 'g2', 'g3', 'w23', 'floor']: print(f"  {k}: {np.mean(top_1_stats[k]):.1f}%")
print(f"  Average Skill Multiplier: {np.mean(top_1_stats['skill']):.2f}x")

print(f"\nTop 10% stats:")
for k in ['g1', 'g2', 'g3', 'w23', 'floor']: print(f"  {k}: {np.mean(top_10_stats[k]):.1f}%")
print(f"  Average Skill Multiplier: {np.mean(top_10_stats['skill']):.2f}x")

print(f"\nBottom 50% stats:")
print(f"  Average Skill Multiplier: {np.mean(bottom_50_skill):.2f}x")

# --- Graphs ---
plt.figure(figsize=(10, 6))
plt.plot(gini_history, color='purple', linewidth=3)
plt.title('Gini Coefficient Over Time (3% Growth, 5% Bets, Meritocracy Included)')
plt.xlabel('Number of Rounds')
plt.ylabel('Wealth Gini Coefficient')
plt.grid(True, alpha=0.3)
plt.axhline(0.85, color='gray', linestyle='--', label='Highly Unequal Real-World Economy (~0.85)')
plt.legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Wealth Distribution Evolution (Meritocracy Model)', fontsize=16, fontweight='bold')
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