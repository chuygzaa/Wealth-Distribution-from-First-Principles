import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def gini(array):
    array = np.sort(array)
    total = np.sum(array)
    if total == 0: return 1.0
    array = array / total
    n = array.shape[0]
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * array)) / n


def run_generational_simulation(estate_tax_rate, trials=10, num_rounds=2000):
    num_agents = 1000
    starting_wealth = 1000.0
    growth_rate = 0.03
    flat_cost_rate = 0.02
    lifestyle_rate = 0.01

    bet_percentage = 0.05
    exposure_rate = 0.20
    market_cap_limit = 0.001  # Maximum single transaction is 0.1% of total system wealth

    metrics = {
        'depth_top10': defaultdict(lambda: {'total': 0, 'stayed': 0, 'skill_sum': 0.0}),
        'skill_top_10_overall': [],
        'skill_1_to_10_overall': [],
        'skill_bottom_50_overall': [],
        'final_gini': []
    }

    print(f"\n==================================================")
    print(f"RUNNING SIMULATION: {estate_tax_rate * 100:.0f}% ESTATE TAX")
    print(f"==================================================")

    for t in range(trials):
        wealth = np.full(num_agents, starting_wealth)
        skills = np.clip(np.random.normal(1.0, 0.2, num_agents), 0.2, 2.0)

        ages = np.arange(num_agents) % 80
        gen1_completed = np.zeros(num_agents, dtype=bool)

        # New Tracker: How many generations have they stayed in the Top 10%?
        top10_dynasty_streak = np.zeros(num_agents, dtype=int)

        if t == 0:
            gini_history = [gini(wealth)]
            wealth_snapshots = {0: wealth.copy()}
            snapshot_rounds = [10, 50, 200, 500,1000,num_rounds]

        for round_num in range(1, num_rounds + 1):
            ages += 1

            # --- GENERATIONAL TURNOVER ---
            dying_mask = ages == 80
            if np.any(dying_mask):
                dying_indices = np.where(dying_mask)[0]
                threshold_top_1 = np.percentile(wealth, 99)
                threshold_top_10 = np.percentile(wealth, 90)

                for idx in dying_indices:
                    streak = top10_dynasty_streak[idx]
                    is_top_1 = wealth[idx] >= threshold_top_1
                    is_top_10 = wealth[idx] >= threshold_top_10

                    # Gen 1 Founder Logic (Must hit Top 1% to start a dynasty)
                    if not gen1_completed[idx]:
                        gen1_completed[idx] = True
                        if is_top_1:
                            # Start the streak! Their heir is Gen 2.
                            top10_dynasty_streak[idx] = 1

                    # Descendant Logic (Only evaluated against the Top 10% threshold)
                    else:
                        if streak > 0:
                            metrics['depth_top10'][streak]['total'] += 1
                            metrics['depth_top10'][streak]['skill_sum'] += skills[idx]

                            if is_top_10:
                                metrics['depth_top10'][streak]['stayed'] += 1
                                top10_dynasty_streak[idx] += 1
                            else:
                                top10_dynasty_streak[idx] = 0

                tax_collected = np.sum(wealth[dying_mask] * estate_tax_rate)
                wealth[dying_mask] *= (1.0 - estate_tax_rate)

                parent_skills = skills[dying_mask]
                skills[dying_mask] = np.clip(np.random.normal(loc=parent_skills, scale=0.15), 0.2, 2.0)

                ages[dying_mask] = 0

                num_newborns = len(dying_indices)
                if num_newborns > 0:
                    baby_bond = tax_collected / num_newborns
                    wealth[dying_mask] += baby_bond

            # --- STEP 1: The Wage ---
            total_wealth = np.sum(wealth)
            new_money = total_wealth * growth_rate
            base_wage = new_money / num_agents

            luck = np.random.rand(num_agents)
            performance_score = skills * luck
            ranks = np.argsort(np.argsort(performance_score))
            wage_multipliers = 0.5 + (ranks / (num_agents - 1)) * (1.5 - 0.5)

            wealth += (base_wage * wage_multipliers)

            # --- STEP 2: Consumption ---
            current_avg_wealth = np.sum(wealth) / num_agents
            flat_cost = current_avg_wealth * flat_cost_rate
            lifestyle_cost = wealth * lifestyle_rate
            intended_consumption = flat_cost + lifestyle_cost

            actual_consumption = np.minimum(wealth, intended_consumption)
            wealth -= actual_consumption
            corporate_revenue = np.sum(actual_consumption)

            safe_total = np.where(np.sum(wealth) == 0, 1e-9, np.sum(wealth))
            wealth += corporate_revenue * (wealth / safe_total)

            # --- STEP 3: The Bets ---
            indices = np.arange(num_agents)
            np.random.shuffle(indices)

            idx_a, idx_b = indices[0::2], indices[1::2]
            w_a, w_b = wealth[idx_a], wealth[idx_b]
            s_a, s_b = skills[idx_a], skills[idx_b]

            liquid_w_a = w_a * exposure_rate
            liquid_w_b = w_b * exposure_rate

            current_total_system_wealth = np.sum(wealth)
            max_bet_cap = current_total_system_wealth * market_cap_limit

            intended_bet = (liquid_w_a + liquid_w_b) * bet_percentage
            actual_bet = np.minimum(intended_bet, np.minimum(liquid_w_a, liquid_w_b))
            actual_bet = np.minimum(actual_bet, max_bet_cap)

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

            if np.max(wealth) > 1e150:
                wealth /= 1e100

            # --- TRACK CLASS SKILL ---
            if round_num % 10 == 0:  # Sample every 10 rounds to save processing power
                t10_thresh = np.percentile(wealth, 90)
                t1_thresh = np.percentile(wealth, 99)
                median_thresh = np.percentile(wealth, 50)

                mask_top_10 = wealth >= t10_thresh
                mask_1_to_10 = (wealth >= t10_thresh) & (wealth < t1_thresh)
                mask_bottom_50 = wealth <= median_thresh

                metrics['skill_top_10_overall'].append(np.mean(skills[mask_top_10]))
                metrics['skill_1_to_10_overall'].append(np.mean(skills[mask_1_to_10]))
                metrics['skill_bottom_50_overall'].append(np.mean(skills[mask_bottom_50]))

            if t == 0:
                gini_history.append(gini(wealth))
                if round_num in snapshot_rounds:
                    wealth_snapshots[round_num] = wealth.copy()

        metrics['final_gini'].append(gini(wealth))

    # --- Output Analytics ---
    print(f"Final Gini Coefficient:      {np.mean(metrics['final_gini']):.3f}")

    print(f"\n--- Class Intelligence Averages ---")
    print(f"Average Skill of Top 10%:       {np.mean(metrics['skill_top_10_overall']):.2f}x")
    print(f"Average Skill of Top 1.1%-10%:  {np.mean(metrics['skill_1_to_10_overall']):.2f}x")
    print(f"Average Skill of Bottom 50%:    {np.mean(metrics['skill_bottom_50_overall']):.2f}x")

    print(f"\n--- Dynasty Survival (Top 1% Founders -> Top 10% Descendants) ---")

    deep_royalty = {'total': 0, 'stayed': 0, 'skill_sum': 0.0}

    for depth in sorted(metrics['depth_top10'].keys()):
        stats = metrics['depth_top10'][depth]
        stay_rate = (stats['stayed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        avg_skill = (stats['skill_sum'] / stats['total']) if stats['total'] > 0 else 0

        generation_label = depth + 1

        if generation_label < 5:
            if stats['total'] > 50:
                print(f"Gen {generation_label} Heirs (Evaluated {stats['total']} lifetimes):")
                print(f"  └─ Average Skill of Cohort:  {avg_skill:.2f}x")
                print(f"  └─ Maintained Top 10%:       {stay_rate:.1f}%")
        else:
            deep_royalty['total'] += stats['total']
            deep_royalty['stayed'] += stats['stayed']
            deep_royalty['skill_sum'] += stats['skill_sum']

    if deep_royalty['total'] > 50:
        stay_rate = deep_royalty['stayed'] / deep_royalty['total'] * 100
        avg_skill_deep = deep_royalty['skill_sum'] / deep_royalty['total']
        print(f"\nGen 5+ Heirs [Deep Royalty] (Evaluated {deep_royalty['total']} lifetimes):")
        print(f"  └─ Average Skill of Cohort:  {avg_skill_deep:.2f}x")
        print(f"  └─ Maintained Top 10%:       {stay_rate:.1f}%")

    print(f"==================================================\n")

    if estate_tax_rate == 1.0:
        plt.figure(figsize=(10, 6))
        plt.plot(gini_history, color='purple', linewidth=3)
        plt.title('Gini Coefficient Over Time (100% Tax Scenario)')
        plt.xlabel('Number of Rounds')
        plt.ylabel('Wealth Gini Coefficient')
        plt.grid(True, alpha=0.3)
        plt.axhline(0.85, color='gray', linestyle='--', label='Real-World Target (~0.85)')
        plt.legend()
        plt.show()

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Wealth Distribution Evolution (100% Tax Scenario)', fontsize=16, fontweight='bold')
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

scenarios = [0.0, 0.40, 1.0]
for tax_rate in scenarios:
    run_generational_simulation(tax_rate, trials=10)