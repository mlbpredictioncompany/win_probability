import pandas as pd
import numpy as np

######

# READ TRANSITION COUNTS FILE
df = pd.read_csv('mlb_transition_counts.csv')
print(df)


# ADD ZERO ROWS TO ENSURE SQUARE MATRICES
all_states = list(set(list(df['game_state'].unique()) + list(df['next_state'].unique())))
zero_rows = [{'game_state': s1, 'next_state': s2, 'runs_scored': r, 'transition_count': 0} for s1 in all_states for s2 in all_states for r in range(0, 5)]
df = pd.concat([df, pd.DataFrame(zero_rows)], axis=0)
print(df)


# ADD ABSORBENT STATE TRANSITIONS (IDENTITY)
absorbent_states = [f'3-{i}' for i in range(0, 4)]
absorbent_rows = [{'game_state': s, 'next_state': s, 'runs_scored': 0, 'transition_count': 1} for s in absorbent_states]
df = pd.concat([df, pd.DataFrame(absorbent_rows)], axis=0)
print(df)


# TRANSITION MATRIX
transitions = pd.pivot_table(data=df, values='transition_count', index='game_state', columns='next_state', fill_value=0, aggfunc='sum')
transitions = transitions / transitions.values.sum(axis=1, keepdims=True)
transitions.fillna(0, inplace=True)
print(transitions)


# REWARD MATRIX
rewards = pd.pivot_table(data=df.loc[df['transition_count'] > 0], values='runs_scored', index='game_state', columns='next_state', fill_value=0)
print(rewards)


# RUNS EXPECTED AFTER N STEPS
n_steps = 100
runs_expected = np.hstack([(np.linalg.matrix_power(transitions.values, i - 1) @ (transitions.values * rewards.values)).sum(axis=1).reshape(-1, 1) for i in range(1, n_steps + 1)]).sum(axis=1)
runs_expected = pd.DataFrame(data=runs_expected, index=transitions.index, columns=['runs_expected'])
print(runs_expected)
