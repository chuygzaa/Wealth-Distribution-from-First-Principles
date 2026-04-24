import random
import numpy as np
import matplotlib.pyplot as plt


def run_simulation(starting_wealth=1000, bet_percentage=0.10, growth_percentage=0.10, floor_percentage=0.10,
                   max_iterations=200):
    """
    Simulates a repeated proportional betting game with economic growth and a social safety net floor.
    """
    wealth_a = starting_wealth
    wealth_b = starting_wealth

    winners = []
    hit_floor_a = False
    hit_floor_b = False

    for i in range(max_iterations):
        # 1. Calculate and cap bets
        total_wealth = wealth_a + wealth_b
        current_bet = total_wealth * bet_percentage
        actual_bet = min(current_bet, wealth_a, wealth_b)

        # 2. Play the round
        prob_a_wins = wealth_a / total_wealth

        if random.random() < prob_a_wins:
            wealth_a += actual_bet
            wealth_b -= actual_bet
            winners.append('A')
        else:
            wealth_a -= actual_bet
            wealth_b += actual_bet
            winners.append('B')

        # 3. Apply Proportional Growth with Dynamic Floor
        total_wealth = wealth_a + wealth_b
        system_growth = total_wealth * growth_percentage

        prop_a = wealth_a / total_wealth
        prop_b = wealth_b / total_wealth

        if prop_a <= floor_percentage:
            hit_floor_a = True
            growth_a = system_growth * floor_percentage
            growth_b = system_growth * (1.0 - floor_percentage)
        elif prop_b <= floor_percentage:
            hit_floor_b = True
            growth_a = system_growth * (1.0 - floor_percentage)
            growth_b = system_growth * floor_percentage
        else:
            growth_a = system_growth * prop_a
            growth_b = system_growth * prop_b

        wealth_a += growth_a
        wealth_b += growth_b

    # --- Process Final Outcomes ---
    ultimate_winner = 'A' if wealth_a > wealth_b else 'B'

    g1_takes_all = (winners[0] == ultimate_winner)

    first_2_winner = winners[0] if (len(winners) >= 2 and winners[0] == winners[1]) else None
    g2_takes_all = (first_2_winner == ultimate_winner) if first_2_winner else None

    first_3_winner = winners[0] if (len(winners) >= 3 and winners[0] == winners[1] == winners[2]) else None
    g3_takes_all = (first_3_winner == ultimate_winner) if first_3_winner else None

    w23 = 'A' if winners[:3].count('A') >= 2 else 'B'
    g2of3_takes_all = (w23 == ultimate_winner)

    someone_hit_floor = hit_floor_a or hit_floor_b
    floor_comeback = False
    if (ultimate_winner == 'A' and hit_floor_a) or (ultimate_winner == 'B' and hit_floor_b):
        floor_comeback = True

    return {
        'g1_win': g1_takes_all,
        'g2_sweep': first_2_winner is not None,
        'g2_win': g2_takes_all,
        'g3_sweep': first_3_winner is not None,
        'g3_win': g3_takes_all,
        'g2of3_win': g2of3_takes_all,
        'floor_hit': someone_hit_floor,
        'floor_comeback': floor_comeback
    }


def run_monte_carlo(trials=1000, starting_wealth=1000, bet_percentage=0.10, growth_percentage=0.10,
                    floor_percentage=0.10, max_iterations=200):
    """
    Aggregates thousands of simulations to find stabilized probability distributions.
    """
    stats = {'g1_wins': 0, 'g2_sweeps': 0, 'g2_wins': 0, 'g3_sweeps': 0, 'g3_wins': 0,
             'g2of3_wins': 0, 'floor_hits': 0, 'floor_comebacks': 0}

    for _ in range(trials):
        outcomes = run_simulation(starting_wealth, bet_percentage, growth_percentage, floor_percentage, max_iterations)

        if outcomes['g1_win']: stats['g1_wins'] += 1

        if outcomes['g2_sweep']:
            stats['g2_sweeps'] += 1
            if outcomes['g2_win']: stats['g2_wins'] += 1

        if outcomes['g3_sweep']:
            stats['g3_sweeps'] += 1
            if outcomes['g3_win']: stats['g3_wins'] += 1

        if outcomes['g2of3_win']: stats['g2of3_wins'] += 1

        if outcomes['floor_hit']:
            stats['floor_hits'] += 1
            if outcomes['floor_comeback']: stats['floor_comebacks'] += 1

    return {
        'g1_prob': (stats['g1_wins'] / trials) * 100,
        'g2_prob': (stats['g2_wins'] / stats['g2_sweeps'] * 100) if stats['g2_sweeps'] > 0 else 0,
        'g3_prob': (stats['g3_wins'] / stats['g3_sweeps'] * 100) if stats['g3_sweeps'] > 0 else 0,
        'g2of3_prob': (stats['g2of3_wins'] / trials) * 100,
        'floor_hit_rate': (stats['floor_hits'] / trials) * 100,
        'comeback_prob': (stats['floor_comebacks'] / stats['floor_hits'] * 100) if stats['floor_hits'] > 0 else 0
    }


def plot_sweep(x_values, results, x_label, title):
    """
    Generates a 2x3 grid of subplots for a given parameter sweep.
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    metrics = [
        ('g1_prob', 'Game 1 Winner Wins All (%)', 'blue'),
        ('g2_prob', '2-Game Sweeper Wins All (%)', 'green'),
        ('g3_prob', '3-Game Sweeper Wins All (%)', 'purple'),
        ('g2of3_prob', '2/3 Early Winner Wins All (%)', 'orange'),
        ('floor_hit_rate', 'Simulations Hitting Floor (%)', 'red'),
        ('comeback_prob', 'Floor Comeback Rate (%)', 'brown')
    ]

    for i, (key, ylabel, color) in enumerate(metrics):
        row = i // 3
        col = i % 3
        y_values = [res[key] for res in results]

        axs[row, col].plot(x_values, y_values, marker='o', color=color, linewidth=2)
        axs[row, col].set_title(ylabel)
        axs[row, col].set_xlabel(x_label)
        axs[row, col].set_ylabel('Probability / Rate (%)')
        axs[row, col].grid(True, alpha=0.3)
        axs[row, col].set_ylim(bottom=0)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()


# ==========================================
# MASTER EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":

    # Base Control Parameters
    trials_per_step = 10000  # <--- Change to 10,000 for maximum resolution
    fixed_bet = 0.10
    fixed_growth = 0.10
    fixed_floor = 0.10

    print(f"--- Initialization ---")
    print(f"Trials per step: {trials_per_step:,}")
    print(f"Base Parameters - Bet: {fixed_bet * 100}%, Growth: {fixed_growth * 100}%, Floor: {fixed_floor * 100}%\n")

    # --- SWEEP 1: BET SIZE ---
    bet_percentages = np.arange(0.01, 0.50, 0.02)
    print("Starting Sweep 1: Varying Bet Size...")
    bet_results = []
    for bet in bet_percentages:
        print(f"  Calculating bet size {bet * 100:.0f}%...")
        res = run_monte_carlo(trials=trials_per_step, bet_percentage=bet, growth_percentage=fixed_growth,
                              floor_percentage=fixed_floor)
        bet_results.append(res)

    print("Generating Graph 1...")
    plot_sweep(bet_percentages * 100, bet_results, 'Bet Size (% of Total Wealth)',
               f'Sweep 1: Impact of Bet Size (Growth: {fixed_growth * 100}%, Floor: {fixed_floor * 100}%)')

    # --- SWEEP 2: SYSTEM GROWTH ---
    growth_percentages = np.arange(0.01, 0.26, 0.01)
    print("\nStarting Sweep 2: Varying System Growth...")
    growth_results = []
    for growth in growth_percentages:
        print(f"  Calculating system growth {growth * 100:.0f}%...")
        res = run_monte_carlo(trials=trials_per_step, bet_percentage=fixed_bet, growth_percentage=growth,
                              floor_percentage=fixed_floor)
        growth_results.append(res)

    print("Generating Graph 2...")
    plot_sweep(growth_percentages * 100, growth_results, 'System Growth (% of Total Wealth)',
               f'Sweep 2: Impact of System Growth (Bet: {fixed_bet * 100}%, Floor: {fixed_floor * 100}%)')

    # --- SWEEP 3: SAFETY NET FLOOR ---
    floor_percentages = np.arange(0.01, 0.31, 0.01)
    print("\nStarting Sweep 3: Varying Safety Net Floor...")
    floor_results = []
    for floor in floor_percentages:
        print(f"  Calculating floor level {floor * 100:.0f}%...")
        res = run_monte_carlo(trials=trials_per_step, bet_percentage=fixed_bet, growth_percentage=fixed_growth,
                              floor_percentage=floor)
        floor_results.append(res)

    print("Generating Graph 3...")
    plot_sweep(floor_percentages * 100, floor_results, 'Safety Net Floor (% of Total Wealth)',
               f'Sweep 3: Impact of Safety Net Floor (Bet: {fixed_bet * 100}%, Growth: {fixed_growth * 100}%)')

    print("\nSimulation complete.")