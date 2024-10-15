import pandas as pd
import numpy as np

######

# READ TRANSITION COUNTS FILE
df = pd.read_csv('mlb_transition_counts.csv')
# df = pd.read_csv('coors_field_transition_counts.csv')
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


# CREATE RANDOM TRANSITION MATRICES FOR EACH MATCHUP
matchup_transitions = []

for i in range(9):
    # GET RANDOM PERTRUBATION COEFFICIENTS
    x = np.random.normal(loc=0, scale=1, size=transitions.values.shape)
    x = np.exp(x)

    # SCALE TRANSFORMATION MATRIX
    transitions_i = np.power(transitions.values, x)
    transitions_i = transitions_i / transitions_i.sum(axis=1, keepdims=True)

    # APPEND
    matchup_transitions.append(transitions_i)


# RUNS EXPECTED AFTER N STEPS
n_steps = 100
prev_transitions = np.identity(transitions.values.shape[0])
runs_expected = []

for i in range(n_steps):
    # SELECT MATRICES
    transitions_i = matchup_transitions[i % 9]

    # GET EXPECTED RUNS
    runs_expected_i = (prev_transitions @ (transitions_i * rewards.values)).sum(axis=1).reshape(-1, 1)
    runs_expected.append(runs_expected_i)

    # UPDATE PREV TRANSITIONS
    prev_transitions = np.matmul(prev_transitions, transitions_i)

# CONSOLIDATE RUNS EXPECTED (SUM STEPS)
runs_expected = np.hstack(runs_expected).sum(axis=1)
runs_expected = pd.DataFrame(data=runs_expected, index=transitions.index, columns=['runs_expected'])
print(runs_expected)


