import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Macroeconomic Physics", layout="wide")
st.title("Macroeconomic Structural Physics Engine")
st.markdown("Wealth Distribution from First Principles: An Agent-Based Simulation of Merit, Luck, and Dynastic Survival.")


# --- MATH & SIMULATION ENGINE ---
def gini(array):
    array = np.sort(np.maximum(0, array))
    total = np.sum(array)
    if total <= 0: return 1.0
    array = array / total
    n = array.shape[0]
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * array)) / n


def safe_mean(lst):
    return np.mean(lst) if len(lst) > 0 else 0.0


def run_generational_simulation(estate_tax_rate, trials, num_rounds):
    num_agents = 1000
    starting_liquid, starting_assets, initial_asset_price = 990.0, 10.0, 1.0
    gdp_growth_rate, asset_appreciation, asset_dividend = 0.03, 0.06, 0.03
    flat_cost_rate, lifestyle_rate = 0.02, 0.01
    bet_percentage, exposure_rate, market_cap_limit = 0.05, 0.20, 0.001

    snapshot_rounds = [10, 50, 100, 250, 500, num_rounds]
    metrics = {rnd: defaultdict(list) for rnd in snapshot_rounds}
    wealth_snapshots = {rnd: [] for rnd in snapshot_rounds}
    history_interval = 10
    history_rounds = list(range(history_interval, num_rounds + 1, history_interval))
    history_metrics = {rnd: defaultdict(list) for rnd in history_rounds}

    for t in range(trials):
        liquid_wealth = np.full(num_agents, starting_liquid)
        assets = np.full(num_agents, starting_assets)
        asset_price = initial_asset_price

        skills = np.clip(np.random.normal(1.0, 0.2, num_agents), 0.2, 2.0)
        ages = np.arange(num_agents) % 80

        gen1_completed = np.zeros(num_agents, dtype=bool)
        dynasty_depth = np.zeros(num_agents, dtype=int)
        dynasty_depth_tracker = defaultdict(lambda: {'total': 0, 'stayed': 0, 'skill_sum': 0.0})

        gen1_bets_played = np.zeros(num_agents, dtype=int)
        gen1_bets_won = np.zeros(num_agents, dtype=int)
        gen1_won_first = np.zeros(num_agents, dtype=bool)
        gen1_won_2_of_3 = np.zeros(num_agents, dtype=bool)

        total_lifetimes, lifetimes_top_1, lifetimes_1_to_10, lifetimes_bottom_50 = (np.zeros(num_agents, dtype=int) for
                                                                                    _ in range(4))


        for round_num in range(1, num_rounds + 1):
            ages += 1
            asset_price *= (1.0 + asset_appreciation)
            # Dividends pay out as liquid cash based on current asset value
            liquid_wealth += (assets * asset_price) * asset_dividend
            total_wealth = liquid_wealth + (assets * asset_price)

            dying_mask = ages == 80
            if round_num in history_rounds:
                t1_thresh = np.percentile(total_wealth, 99)
                t10_thresh = np.percentile(total_wealth, 90)
                med_thresh = np.percentile(total_wealth, 50)

                t1_mask = total_wealth >= t1_thresh
                t10_mask = total_wealth >= t10_thresh
                b50_mask = total_wealth <= med_thresh

                hm = history_metrics[round_num]
                hm['gini_total'].append(gini(total_wealth))
                hm['gini_liquid'].append(gini(liquid_wealth))
                hm['gini_asset'].append(gini(assets))

                hm['skill_top_1'].append(safe_mean(skills[t1_mask]))
                hm['skill_1_to_10'].append(safe_mean(skills[(t10_mask) & (~t1_mask)]))
                hm['skill_bottom_50'].append(safe_mean(skills[b50_mask]))

                t1_tot = max(1, np.sum(t1_mask))
                t10_tot = max(1, np.sum(t10_mask))

                hm['fl_t1_1st'].append(np.sum(gen1_won_first[t1_mask]) / t1_tot * 100)
                hm['fl_t1_2of3'].append(np.sum(gen1_won_2_of_3[t1_mask]) / t1_tot * 100)
                hm['fl_t10_1st'].append(np.sum(gen1_won_first[t10_mask]) / t10_tot * 100)
                hm['fl_t10_2of3'].append(np.sum(gen1_won_2_of_3[t10_mask]) / t10_tot * 100)

                dumb_mask, smart_mask = skills < 1.0, skills >= 1.4
                hm['dumb_t1'].append(np.sum(t1_mask & dumb_mask) / t1_tot * 100)
                hm['dumb_t10'].append(np.sum(t10_mask & dumb_mask) / t10_tot * 100)
                hm['smart_b50'].append(np.sum(b50_mask & smart_mask) / max(1, np.sum(b50_mask)) * 100)

                def get_ret(idx):
                    stats = dynasty_depth_tracker[idx]
                    return (stats['stayed'] / stats['total'] * 100) if stats['total'] > 0 else 0.0

                deep_tot, deep_stay = 0, 0
                for d in dynasty_depth_tracker.keys():
                    if d >= 4:
                        deep_tot += dynasty_depth_tracker[d]['total']
                        deep_stay += dynasty_depth_tracker[d]['stayed']

                hm['ret_gen2'].append(get_ret(1))
                hm['ret_gen3'].append(get_ret(2))
                hm['ret_gen4'].append(get_ret(3))
                hm['ret_gen5'].append((deep_stay / deep_tot * 100) if deep_tot > 0 else 0.0)

                safe_life_t1 = np.maximum(1, total_lifetimes[t1_mask])
                hm['anc_t1_t1'].append(np.mean(lifetimes_top_1[t1_mask] / safe_life_t1) * 100)
                hm['anc_t1_t10'].append(np.mean(lifetimes_1_to_10[t1_mask] / safe_life_t1) * 100)
                hm['anc_t1_b50'].append(np.mean(lifetimes_bottom_50[t1_mask] / safe_life_t1) * 100)

            if np.any(dying_mask):
                dying_indices = np.where(dying_mask)[0]
                threshold_top_1 = np.percentile(total_wealth, 99)
                threshold_top_10 = np.percentile(total_wealth, 90)
                threshold_med = np.percentile(total_wealth, 50)

                for idx in dying_indices:
                    is_top_1 = total_wealth[idx] >= threshold_top_1
                    is_top_10 = total_wealth[idx] >= threshold_top_10
                    is_bottom_50 = total_wealth[idx] <= threshold_med

                    total_lifetimes[idx] += 1
                    if is_top_1:
                        lifetimes_top_1[idx] += 1
                    elif is_top_10:
                        lifetimes_1_to_10[idx] += 1
                    if is_bottom_50: lifetimes_bottom_50[idx] += 1

                    depth = dynasty_depth[idx]
                    if not gen1_completed[idx]:
                        gen1_completed[idx] = True
                        if is_top_1: dynasty_depth[idx] = 1
                    else:
                        if depth > 0:
                            dynasty_depth_tracker[depth]['total'] += 1
                            if is_top_10:
                                dynasty_depth_tracker[depth]['stayed'] += 1
                                dynasty_depth[idx] += 1
                            else:
                                dynasty_depth[idx] = 0

                liquid_tax = np.sum(liquid_wealth[dying_mask] * estate_tax_rate)
                asset_tax = np.sum(assets[dying_mask] * estate_tax_rate)
                liquid_wealth[dying_mask] *= (1.0 - estate_tax_rate)
                assets[dying_mask] *= (1.0 - estate_tax_rate)

                parent_skills = skills[dying_mask]
                skills[dying_mask] = np.clip(np.random.normal(loc=parent_skills, scale=0.15), 0.2, 2.0)
                ages[dying_mask] = 0

                num_newborns = len(dying_indices)
                if num_newborns > 0:
                    liquid_wealth[dying_mask] += (liquid_tax / num_newborns)
                    assets[dying_mask] += (asset_tax / num_newborns)

            # Wages, Consumption & Bets
            total_liquid = np.sum(liquid_wealth)
            new_money = total_liquid * gdp_growth_rate
            base_wage = new_money / num_agents

            luck = np.random.rand(num_agents)
            ranks = np.argsort(np.argsort(skills * luck))
            wage_multipliers = 0.5 + (ranks / (num_agents - 1)) * (1.5 - 0.5)
            liquid_wealth += (base_wage * wage_multipliers)

            total_wealth = liquid_wealth + (assets * asset_price)
            current_avg_liquid = np.sum(liquid_wealth) / num_agents
            intended_consumption = (current_avg_liquid * flat_cost_rate) + (total_wealth * lifestyle_rate)

            shortfall = np.maximum(0, intended_consumption - liquid_wealth)
            assets_to_sell = shortfall / asset_price
            actual_assets_sold = np.minimum(assets, assets_to_sell)

            assets -= actual_assets_sold
            liquid_wealth += (actual_assets_sold * asset_price)
            actual_consumption = np.minimum(liquid_wealth, intended_consumption)
            liquid_wealth -= actual_consumption

            total_asset_value = assets * asset_price
            safe_asset_total = np.where(np.sum(total_asset_value) == 0, 1e-9, np.sum(total_asset_value))
            liquid_wealth += np.sum(actual_consumption) * (total_asset_value / safe_asset_total)

            indices = np.arange(num_agents)
            np.random.shuffle(indices)
            idx_a, idx_b = indices[0::2], indices[1::2]

            liquid_exposure_a, liquid_exposure_b = liquid_wealth[idx_a] * exposure_rate, liquid_wealth[
                idx_b] * exposure_rate
            asset_exposure_a, asset_exposure_b = assets[idx_a] * exposure_rate, assets[idx_b] * exposure_rate

            max_liquid_cap = np.sum(liquid_wealth) * market_cap_limit
            max_asset_cap = np.sum(assets) * market_cap_limit

            liquid_bet = np.minimum((liquid_exposure_a + liquid_exposure_b) * bet_percentage,
                                    np.minimum(liquid_exposure_a, liquid_exposure_b))
            liquid_bet = np.minimum(liquid_bet, max_liquid_cap)
            asset_bet = np.minimum((asset_exposure_a + asset_exposure_b) * bet_percentage,
                                   np.minimum(asset_exposure_a, asset_exposure_b))
            asset_bet = np.minimum(asset_bet, max_asset_cap)

            advantage_a, advantage_b = total_wealth[idx_a] * skills[idx_a], total_wealth[idx_b] * skills[idx_b]
            safe_total_adv = np.where((advantage_a + advantage_b) == 0, 1e-9, advantage_a + advantage_b)
            a_wins = np.random.rand(len(idx_a)) < (advantage_a / safe_total_adv)
            b_wins = ~a_wins

            def track_gen1(idxs, wins):
                mask = ~gen1_completed[idxs]
                idx_gen1, wins_gen1 = idxs[mask], wins[mask]
                gen1_bets_played[idx_gen1] += 1
                gen1_bets_won[idx_gen1[wins_gen1]] += 1
                mask_first = gen1_bets_played[idx_gen1] == 1
                gen1_won_first[idx_gen1[mask_first]] = wins_gen1[mask_first]
                mask_third = gen1_bets_played[idx_gen1] == 3
                gen1_won_2_of_3[idx_gen1[mask_third]] = (gen1_bets_won[idx_gen1[mask_third]] >= 2)

            track_gen1(idx_a, a_wins)
            track_gen1(idx_b, b_wins)

            liquid_wealth[idx_a[a_wins]] += liquid_bet[a_wins]
            liquid_wealth[idx_b[a_wins]] -= liquid_bet[a_wins]
            liquid_wealth[idx_b[b_wins]] += liquid_bet[b_wins]
            liquid_wealth[idx_a[b_wins]] -= liquid_bet[b_wins]

            assets[idx_a[a_wins]] += asset_bet[a_wins]
            assets[idx_b[a_wins]] -= asset_bet[a_wins]
            assets[idx_b[b_wins]] += asset_bet[b_wins]
            assets[idx_a[b_wins]] -= asset_bet[b_wins]

            total_wealth = liquid_wealth + (assets * asset_price)

            if round_num in snapshot_rounds:
                wealth_snapshots[round_num].extend(total_wealth.tolist())

                t1_thresh, t10_thresh, med_thresh = np.percentile(total_wealth, 99), np.percentile(total_wealth,
                                                                                                   90), np.percentile(
                    total_wealth, 50)
                t1_mask, t10_mask, b50_mask = total_wealth >= t1_thresh, total_wealth >= t10_thresh, total_wealth <= med_thresh

                m = metrics[round_num]
                m['gini_total'].append(gini(total_wealth))
                m['gini_liquid'].append(gini(liquid_wealth))
                m['gini_asset'].append(gini(assets))
                m['skill_top_1'].append(np.mean(skills[t1_mask]))
                m['skill_1_to_10'].append(np.mean(skills[(t10_mask) & (~t1_mask)]))
                m['skill_bottom_50'].append(np.mean(skills[b50_mask]))

                t1_tot, t10_tot = max(1, np.sum(t1_mask)), max(1, np.sum(t10_mask))
                m['fl_t1_1st'].append(np.sum(gen1_won_first[t1_mask]) / t1_tot * 100)
                m['fl_t1_2of3'].append(np.sum(gen1_won_2_of_3[t1_mask]) / t1_tot * 100)
                m['fl_t10_1st'].append(np.sum(gen1_won_first[t10_mask]) / t10_tot * 100)
                m['fl_t10_2of3'].append(np.sum(gen1_won_2_of_3[t10_mask]) / t10_tot * 100)

                dumb_mask, smart_mask = skills < 1.0, skills >= 1.4
                m['dumb_t1'].append(np.sum(t1_mask & dumb_mask) / t1_tot * 100)
                m['dumb_t10'].append(np.sum(t10_mask & dumb_mask) / t10_tot * 100)
                m['smart_b50'].append(np.sum(b50_mask & smart_mask) / max(1, np.sum(b50_mask)) * 100)

                def get_ret(idx):
                    stats = dynasty_depth_tracker[idx]
                    return (stats['stayed'] / stats['total'] * 100) if stats['total'] > 0 else 0.0

                deep_tot, deep_stay = 0, 0
                for d in dynasty_depth_tracker.keys():
                    if d >= 4:
                        deep_tot += dynasty_depth_tracker[d]['total']
                        deep_stay += dynasty_depth_tracker[d]['stayed']
                m['ret_gen2'].append(get_ret(1))
                m['ret_gen3'].append(get_ret(2))
                m['ret_gen4'].append(get_ret(3))
                m['ret_gen5'].append((deep_stay / deep_tot * 100) if deep_tot > 0 else 0.0)

                safe_life_t1 = np.maximum(1, total_lifetimes[t1_mask])
                m['anc_t1_t1'].append(np.mean(lifetimes_top_1[t1_mask] / safe_life_t1) * 100)
                m['anc_t1_t10'].append(np.mean(lifetimes_1_to_10[t1_mask] / safe_life_t1) * 100)
                m['anc_t1_b50'].append(np.mean(lifetimes_bottom_50[t1_mask] / safe_life_t1) * 100)

    results_by_round = {}
    for rnd in snapshot_rounds:
        m = metrics[rnd]
        results_by_round[rnd] = {
            'Gini (Total)': safe_mean(m['gini_total']),
            'Gini (Liquid)': safe_mean(m['gini_liquid']),
            'Gini (Asset)': safe_mean(m['gini_asset']),
            'Skill (Top 1%)': safe_mean(m['skill_top_1']),
            'Skill (Top 10%)': safe_mean(m['skill_1_to_10']),
            'Skill (Bot 50%)': safe_mean(m['skill_bottom_50']),
            'Founder Luck T1 (1st)': safe_mean(m['fl_t1_1st']),
            'Founder Luck T1 (2of3)': safe_mean(m['fl_t1_2of3']),
            'Founder Luck T10 (1st)': safe_mean(m['fl_t10_1st']),
            'Founder Luck T10 (2of3)': safe_mean(m['fl_t10_2of3']),
            'Dumb Rich (Top 1%)': safe_mean(m['dumb_t1']),
            'Dumb Rich (Top 10%)': safe_mean(m['dumb_t10']),
            'Trapped Genius (Bot 50%)': safe_mean(m['smart_b50']),
            'Retention (Gen 2)': safe_mean(m['ret_gen2']),
            'Retention (Gen 3)': safe_mean(m['ret_gen3']),
            'Retention (Gen 4)': safe_mean(m['ret_gen4']),
            'Retention (Gen 5+)': safe_mean(m['ret_gen5']),
            'Ancestry T1 (in T1)': safe_mean(m['anc_t1_t1']),
            'Ancestry T1 (in T10)': safe_mean(m['anc_t1_t10']),
            'Ancestry T1 (in Bot50)': safe_mean(m['anc_t1_b50']),
        }
    # Process history metrics into a DataFrame
    history_results = []
    for rnd in history_rounds:
        m = history_metrics[rnd]
        history_results.append({
            'Round': rnd,
            'Gini (Total)': safe_mean(m['gini_total']),
            'Gini (Liquid)': safe_mean(m['gini_liquid']),
            'Gini (Asset)': safe_mean(m['gini_asset']),
            'Skill (Top 1%)': safe_mean(m['skill_top_1']),
            'Skill (Top 10%)': safe_mean(m['skill_1_to_10']),
            'Skill (Bot 50%)': safe_mean(m['skill_bottom_50']),
            'Founder Luck T1 (1st)': safe_mean(m['fl_t1_1st']),
            'Founder Luck T1 (2of3)': safe_mean(m['fl_t1_2of3']),
            'Founder Luck T10 (1st)': safe_mean(m['fl_t10_1st']),
            'Founder Luck T10 (2of3)': safe_mean(m['fl_t10_2of3']),
            'Dumb Rich (Top 1%)': safe_mean(m['dumb_t1']),
            'Dumb Rich (Top 10%)': safe_mean(m['dumb_t10']),
            'Trapped Genius (Bot 50%)': safe_mean(m['smart_b50']),
            'Retention (Gen 2)': safe_mean(m['ret_gen2']),
            'Retention (Gen 3)': safe_mean(m['ret_gen3']),
            'Retention (Gen 4)': safe_mean(m['ret_gen4']),
            'Retention (Gen 5+)': safe_mean(m['ret_gen5']),
            'Ancestry T1 (in T1)': safe_mean(m['anc_t1_t1']),
            'Ancestry T1 (in T10)': safe_mean(m['anc_t1_t10']),
            'Ancestry T1 (in Bot50)': safe_mean(m['anc_t1_b50']),
        })

    return results_by_round, wealth_snapshots, history_results  # Ensure you return this!



# --- CACHED SWEEP EXECUTION ---
# This decorator ensures Streamlit runs this exact function once, saves it to memory, and never recalculates unless there's a change in trials/rounds.
@st.cache_data(show_spinner="Running 1000-year macroeconomic simulation (This takes a moment)...")
# --- CACHED SWEEP EXECUTION ---
@st.cache_data(show_spinner="Running 1000-year macroeconomic simulation (This takes a moment)...")
def generate_sweep_data(trials, rounds):
    tax_scenarios = np.linspace(0.0, 1.0, 11)
    all_results = []
    all_snapshots = {}
    all_history = []  # Initialize array to catch the new history data

    for tax in tax_scenarios:
        # Catch all THREE returned variables
        res_by_round, snapshots, history_results = run_generational_simulation(tax, trials=trials, num_rounds=rounds)

        # 1. Process Main Metrics
        for rnd, metrics in res_by_round.items():
            metrics['Tax Rate Num'] = tax * 100
            metrics['Round'] = rnd
            all_results.append(metrics)

        # 2. Store Snapshots
        all_snapshots[round(tax, 1)] = snapshots

        # 3. Process History Metrics (Tag each row with its specific tax rate!)
        for h_row in history_results:
            h_row['Tax Rate Num'] = tax * 100
            all_history.append(h_row)

    # Return 3 items, converting the lists to Pandas DataFrames
    return pd.DataFrame(all_results), all_snapshots, pd.DataFrame(all_history)


# --- UI SIDEBAR ---
st.sidebar.header("Simulation Parameters")
st.sidebar.write(
    "Configure the engine. Increasing trials provides smoother statistical curves but takes longer.")
trials = st.sidebar.slider("Trials per Tax Bracket", min_value=1, max_value=100, value=10, step=5)
rounds = 1000

# Fetch Data (Triggers the cached function)
df, all_snapshots, df_history = generate_sweep_data(trials, rounds)

# --- UI MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Macroeconomic Trends (KPIs)", "Phase Space Explorer (Histograms)", "Raw Data Matrix", "Macroeconomic Trends Over Time (KPIs)"])

with tab1:
    st.subheader("Dynamic Macroeconomic Trends")
    st.write("Slide through the rounds to watch the structural physics of the economy evolve over time.")

    # Interactive Round Slider for the line charts
    available_rounds = sorted(df['Round'].unique())
    selected_round = st.select_slider("Select Simulation Round Snapshot", options=available_rounds, value=1000)

    # Filter dataframe
    d = df[df['Round'] == selected_round]

    # Matplotlib Grid
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))

    axs[0, 0].plot(d['Tax Rate Num'], d['Gini (Total)'], marker='o', label='Total Gini', color='purple')
    axs[0, 0].plot(d['Tax Rate Num'], d['Gini (Liquid)'], marker='^', label='Liquid Gini', color='blue', linestyle=':')
    axs[0, 0].plot(d['Tax Rate Num'], d['Gini (Asset)'], marker='s', label='Asset Gini', color='red', linestyle='--')
    axs[0, 0].set_title('1. Wealth Distribution (Gini)')
    axs[0, 0].set_ylim([0.2, 1.05])
    axs[0, 0].grid(True, alpha=0.3);
    axs[0, 0].legend()

    axs[0, 1].plot(d['Tax Rate Num'], d['Skill (Top 1%)'], marker='o', label='Top 1%', color='gold')
    axs[0, 1].plot(d['Tax Rate Num'], d['Skill (Top 10%)'], marker='s', label='Top 1.1-10%', color='green')
    axs[0, 1].plot(d['Tax Rate Num'], d['Skill (Bot 50%)'], marker='^', label='Bottom 50%', color='gray')
    axs[0, 1].set_title('2. Average Class Intelligence')
    axs[0, 1].set_ylim([0.6, 2.0])
    axs[0, 1].grid(True, alpha=0.3);
    axs[0, 1].legend()

    axs[0, 2].plot(d['Tax Rate Num'], d['Founder Luck T1 (1st)'], marker='o', label='T1 (Won 1st Bet)', color='gold')
    axs[0, 2].plot(d['Tax Rate Num'], d['Founder Luck T1 (2of3)'], marker='o', label='T1 (Won 2 of 3)',
                   color='darkorange', linestyle='--')
    axs[0, 2].plot(d['Tax Rate Num'], d['Founder Luck T10 (1st)'], marker='s', label='T10 (Won 1st Bet)', color='green')
    axs[0, 2].plot(d['Tax Rate Num'], d['Founder Luck T10 (2of3)'], marker='s', label='T10 (Won 2 of 3)',
                   color='darkgreen', linestyle='--')
    axs[0, 2].axhline(50, color='black', linestyle=':', alpha=0.5)
    axs[0, 2].set_title("3. Founder's Luck (Gen 1 Correlation)")
    axs[0, 2].set_ylim([30, 85])
    axs[0, 2].grid(True, alpha=0.3);
    axs[0, 2].legend()

    axs[1, 0].plot(d['Tax Rate Num'], d['Dumb Rich (Top 1%)'], marker='o', label='Top 1% (< 1.0x Skill)', color='red')
    axs[1, 0].plot(d['Tax Rate Num'], d['Dumb Rich (Top 10%)'], marker='s', label='Top 10% (< 1.0x Skill)',
                   color='darkred', linestyle='--')
    axs[1, 0].plot(d['Tax Rate Num'], d['Trapped Genius (Bot 50%)'], marker='^', label='Bot 50% (> 1.4x Skill)',
                   color='blue')
    axs[1, 0].set_title('4. The Merit Mismatch')
    axs[1, 0].set_ylim([-5, 50])
    axs[1, 0].grid(True, alpha=0.3);
    axs[1, 0].legend()

    axs[1, 1].plot(d['Tax Rate Num'], d['Retention (Gen 2)'], marker='o', label='Gen 2', color='lightgray')
    axs[1, 1].plot(d['Tax Rate Num'], d['Retention (Gen 3)'], marker='o', label='Gen 3', color='darkgray')
    axs[1, 1].plot(d['Tax Rate Num'], d['Retention (Gen 4)'], marker='o', label='Gen 4', color='gray')
    axs[1, 1].plot(d['Tax Rate Num'], d['Retention (Gen 5+)'], marker='o', label='Gen 5+', color='black', linewidth=3)
    axs[1, 1].set_title('5. Dynasty Retention (Stayed in Top 10%)')
    axs[1, 1].set_ylim([-5, 105])
    axs[1, 1].grid(True, alpha=0.3);
    axs[1, 1].legend()

    axs[1, 2].plot(d['Tax Rate Num'], d['Ancestry T1 (in T1)'], marker='o', label='Time in Top 1%', color='gold')
    axs[1, 2].plot(d['Tax Rate Num'], d['Ancestry T1 (in T10)'], marker='s', label='Time in Top 1.1-10%', color='green')
    axs[1, 2].plot(d['Tax Rate Num'], d['Ancestry T1 (in Bot50)'], marker='^', label='Time in Bottom 50%', color='gray')
    axs[1, 2].set_title('6. Ancestry of Final Top 1% Bloodlines')
    axs[1, 2].set_ylim([-5, 105])
    axs[1, 2].grid(True, alpha=0.3);
    axs[1, 2].legend()

    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Phase Space Explorer (Logarithmic Wealth Distribution)")
    st.write("Hold the tax rate constant to see how the mathematical floor collapses the middle class over time.")

    selected_tax = st.select_slider("Select Estate Tax Rate", options=np.round(np.linspace(0.0, 1.0, 11), 1), value=0.0)

    fig_hist, axs_hist = plt.subplots(2, 3, figsize=(20, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    snapshots = all_snapshots[selected_tax]
    for idx, (rnd, ax) in enumerate(zip(available_rounds, axs_hist.flatten())):
        ax.hist(snapshots[rnd], bins=40, color=colors[idx], edgecolor='black', alpha=0.7)
        ax.set_title(f'Round {rnd}')
        ax.set_xlabel('Total Wealth')
        ax.set_ylabel('Agents (Log)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_hist)

with tab3:
    st.subheader("Raw Data Matrix")
    st.write("The complete dataset for the selected round.")
    formatted_df = d.drop(columns=['Round']).set_index('Tax Rate Num').round(2)
    st.dataframe(formatted_df, use_container_width=True)

with tab4:
    st.subheader("Temporal KPI History (Evolution over 1000 Rounds)")
    st.write("Hold the tax rate constant to observe the exact generational timeline of the macroeconomic phase shift.")

    # Slider for Estate Tax
    history_tax = st.select_slider("Select Estate Tax Rate for Timeline", options=np.round(np.linspace(0.0, 1.0, 11), 1), value=0.0)

    # Filter the history dataframe to the selected tax rate
    dh = df_history[df_history['Tax Rate Num'] == (history_tax * 100)]

    fig_hist2, axs_h2 = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Gini History
    axs_h2[0, 0].plot(dh['Round'], dh['Gini (Total)'], label='Total Gini', color='purple', linewidth=2)
    axs_h2[0, 0].plot(dh['Round'], dh['Gini (Liquid)'], label='Liquid Gini', color='blue', linestyle=':')
    axs_h2[0, 0].plot(dh['Round'], dh['Gini (Asset)'], label='Asset Gini', color='red', linestyle='--')
    axs_h2[0, 0].set_title('1. Wealth Distribution History (Gini)')
    axs_h2[0, 0].set_ylim([0.2, 1.05])
    axs_h2[0, 0].grid(True, alpha=0.3)
    axs_h2[0, 0].legend()

    # 2. Average Class Intelligence History
    axs_h2[0, 1].plot(dh['Round'], dh['Skill (Top 1%)'], label='Top 1%', color='gold', linewidth=2)
    axs_h2[0, 1].plot(dh['Round'], dh['Skill (Top 10%)'], label='Top 1.1-10%', color='green')
    axs_h2[0, 1].plot(dh['Round'], dh['Skill (Bot 50%)'], label='Bottom 50%', color='gray')
    axs_h2[0, 1].set_title('2. Class Intelligence History')
    axs_h2[0, 1].set_ylim([0.6, 2.0])
    axs_h2[0, 1].grid(True, alpha=0.3)
    axs_h2[0, 1].legend()

    # 3. Founder's Luck History
    axs_h2[0, 2].plot(dh['Round'], dh['Founder Luck T1 (1st)'], label='T1 (Won 1st Bet)', color='gold', linewidth=2)
    axs_h2[0, 2].plot(dh['Round'], dh['Founder Luck T1 (2of3)'], label='T1 (Won 2 of 3)', color='darkorange', linestyle='--')
    axs_h2[0, 2].axhline(50, color='black', linestyle=':', alpha=0.5)
    axs_h2[0, 2].set_title("3. Founder's Luck History")
    axs_h2[0, 2].set_ylim([30, 85])
    axs_h2[0, 2].grid(True, alpha=0.3)
    axs_h2[0, 2].legend()

    # 4. Merit Mismatch History
    axs_h2[1, 0].plot(dh['Round'], dh['Dumb Rich (Top 1%)'], label='Top 1% (< 1.0x Skill)', color='red', linewidth=2)
    axs_h2[1, 0].plot(dh['Round'], dh['Dumb Rich (Top 10%)'], label='Top 10% (< 1.0x Skill)', color='darkred', linestyle='--')
    axs_h2[1, 0].plot(dh['Round'], dh['Trapped Genius (Bot 50%)'], label='Bot 50% (> 1.4x Skill)', color='blue')
    axs_h2[1, 0].set_title('4. Merit Mismatch History')
    axs_h2[1, 0].set_ylim([-5, 50])
    axs_h2[1, 0].grid(True, alpha=0.3)
    axs_h2[1, 0].legend()

    # 5. Dynasty Retention History
    axs_h2[1, 1].plot(dh['Round'], dh['Retention (Gen 2)'], label='Gen 2', color='lightgray', linewidth=2)
    axs_h2[1, 1].plot(dh['Round'], dh['Retention (Gen 3)'], label='Gen 3', color='darkgray')
    axs_h2[1, 1].plot(dh['Round'], dh['Retention (Gen 4)'], label='Gen 4', color='gray')
    axs_h2[1, 1].plot(dh['Round'], dh['Retention (Gen 5+)'], label='Gen 5+', color='black', linewidth=3)
    axs_h2[1, 1].set_title('5. Dynasty Retention History')
    axs_h2[1, 1].set_ylim([-5, 105])
    axs_h2[1, 1].grid(True, alpha=0.3)
    axs_h2[1, 1].legend()

    # 6. Ancestry Trajectory History
    axs_h2[1, 2].plot(dh['Round'], dh['Ancestry T1 (in T1)'], label='Time in Top 1%', color='gold', linewidth=2)
    axs_h2[1, 2].plot(dh['Round'], dh['Ancestry T1 (in T10)'], label='Time in Top 1.1-10%', color='green')
    axs_h2[1, 2].plot(dh['Round'], dh['Ancestry T1 (in Bot50)'], label='Time in Bottom 50%', color='gray')
    axs_h2[1, 2].set_title('6. Ancestry Trajectory History')
    axs_h2[1, 2].set_ylim([-5, 105])
    axs_h2[1, 2].grid(True, alpha=0.3)
    axs_h2[1, 2].legend()

    plt.tight_layout()
    st.pyplot(fig_hist2)