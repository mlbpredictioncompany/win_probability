import pandas as pd
import numpy as np
from scipy import sparse

######

# READ TRANSITION COUNTS FILE
df = pd.read_csv('mlb_transition_counts.csv')
# df = pd.read_csv('coors_field_transition_counts.csv')
print(df)


# CLEAN END-OF-INNING STATES
df['inning_end'] = np.where(df['next_state'].isin(['3-0', '3-1', '3-2', '3-3']), 1, 0)
df['next_state'] = np.where(df['inning_end'] == 1, '0-0-0-0', df['next_state'])
print(df)


# ADD INNING/SCORE TOTAL TO GAME STATES
max_score = 20
df_list = []

for inning in range(1, 10):
    for score in range(max_score + 1):
        df_i = df.copy()

        # INNING CHANGES
        df_i['inning_before'] = inning
        df_i['inning_after'] = np.where(df['inning_end'] == 0, df_i['inning_before'], df_i['inning_before'] + 1)

        # SCORE CHANGES
        df_i['score_before'] = score
        df_i['score_after'] = df_i['score_before'] + df_i['runs_scored']
        df_i['score_after'] = np.where(df_i['score_after'] > max_score, max_score, df_i['score_after'])

        # GAME STATES
        df_i['game_state'] = df_i['inning_before'].astype(str) + '-' + df_i['game_state'] + '-' + df_i['score_before'].astype(str)
        df_i['next_state'] = df_i['inning_after'].astype(str) + '-' + df_i['next_state'] + '-' + df_i['score_after'].astype(str)

        df_list.append(df_i)

df = pd.concat(df_list).reset_index(drop=True)
print(df)


# ADD INITIAL/ABSORBENT STATE TRANSITIONS
initial_states = ['1-0-0-0-0-0']
absorbent_states = [f'10-0-0-0-0-{i}' for i in range(max_score + 1)]

initial_rows = [{'game_state': s, 'next_state': s, 'runs_scored': 0, 'transition_count': 0} for s in initial_states]
absorbent_rows = [{'game_state': s, 'next_state': s, 'runs_scored': 0, 'transition_count': 1} for s in absorbent_states]

df = pd.concat([df, pd.DataFrame(absorbent_rows), pd.DataFrame(initial_rows)], axis=0).reset_index(drop=True)
print(df)


# TRANSITION MATRIX
transitions = pd.pivot_table(data=df, values='transition_count', index='game_state', columns='next_state', fill_value=0, aggfunc='sum')
transitions = transitions / transitions.values.sum(axis=1, keepdims=True)
transitions.fillna(0, inplace=True)
print(transitions)


### UNINFORMED

# FINAL STATES
n_steps = 100
final_states = np.linalg.matrix_power(transitions.values, n_steps)
final_states = pd.DataFrame(data=final_states, index=transitions.index, columns=transitions.columns)[absorbent_states]
final_states.columns = [int(x.split('-')[-1]) for x in final_states.columns]
print(final_states)


# MELT PIVOT AND RETAIN ONLY POSSIBLE STATES
final_states = pd.melt(final_states, ignore_index=False, var_name='runs_scored', value_name='probability').reset_index()
final_states = final_states.loc[final_states['probability'] > 0].copy()
final_states[['inning', 'outs', 'runner_1b', 'runner_2b', 'runner_3b', 'team_score']] = np.array(final_states['game_state'].apply(lambda x: np.array(x.split('-'))).to_list()).astype(int)
print(final_states)


# DUPLICATE START-OF-INNING ROWS TO ACCOUNT FOR BATTING AND PITCHING STATES
final_states_batting = final_states.copy()
final_states_batting['bat_flag'] = 1

final_states_pitching = final_states.loc[final_states['outs'] + final_states['runner_1b'] + final_states['runner_2b'] + final_states['runner_3b'] == 0].copy()
final_states_pitching['bat_flag'] = 0

final_states = pd.concat([final_states_batting, final_states_pitching], axis=0).reset_index(drop=True)
final_states['pitch_flag'] = 1 - final_states['bat_flag']
print(final_states)


# DUPLICATE HOME/AWAY
final_states_home = final_states.copy()
final_states_home.rename(columns={'team_score': 'home_score'}, inplace=True)
final_states_home['half_inning'] = np.where(final_states_home['bat_flag'] == 1, 'bottom', 'top')
final_states_home['effective_inning'] = final_states_home['inning']
final_states_home.set_index('effective_inning', inplace=True)

final_states_away = final_states.copy()
final_states_away.rename(columns={'team_score': 'away_score'}, inplace=True)
final_states_away['half_inning'] = np.where(final_states_away['bat_flag'] == 1, 'top', 'bottom')
final_states_away['effective_inning'] = np.where(final_states_away['pitch_flag'] == 1, final_states_away['inning'] - 1, final_states_away['inning'])
final_states_away.set_index('effective_inning', inplace=True)


# JOIN STATES
home_batting = final_states_home.loc[final_states_home['bat_flag'] == 1][['half_inning', 'inning', 'outs', 'runner_1b', 'runner_2b', 'runner_3b', 'home_score', 'runs_scored', 'probability']]
home_pitching = final_states_home.loc[final_states_home['pitch_flag'] == 1][['home_score', 'runs_scored', 'probability']]

away_batting = final_states_away.loc[final_states_away['bat_flag'] == 1][['half_inning', 'inning', 'outs', 'runner_1b', 'runner_2b', 'runner_3b', 'away_score', 'runs_scored', 'probability']]
away_pitching = final_states_away.loc[final_states_away['pitch_flag'] == 1][['away_score', 'runs_scored', 'probability']]

joined_top = home_pitching.join(away_batting, lsuffix='_home', rsuffix='_away').reset_index(drop=True)
joined_bottom = home_batting.join(away_pitching, lsuffix='_home', rsuffix='_away').reset_index(drop=True)
joined = pd.concat([joined_top, joined_bottom], axis=0)
joined = joined.loc[joined['inning'] < 10].copy()
print(joined)




# ### INFORMED

# # CONVERT TO SPARSE
# transitions_sparse = sparse.csr_matrix(transitions.values)


# # CREATE RANDOM TRANSITION MATRICES FOR EACH TEAM/MATCHUP
# matchup_transitions_home = []
# matchup_transitions_away = []

# for i in range(100):
#     # GET RANDOM PERTRUBATION COEFFICIENTS
#     x_home = np.random.normal(loc=0, scale=1)
#     x_home = np.exp(x_home)

#     x_away = np.random.normal(loc=0, scale=1)
#     x_away = np.exp(x_away)

#     # SCALE TRANSFORMATION MATRICES
#     transitions_home = transitions_sparse.power(x_home)
#     transitions_away = transitions_sparse.power(x_away)

#     transitions_home = transitions_home / transitions_home.sum(axis=1)
#     transitions_away = transitions_away / transitions_away.sum(axis=1)

#     # APPEND
#     matchup_transitions_home.append(transitions_home)
#     matchup_transitions_away.append(transitions_away)


# # FINAL STATES AFTER N STEPS FOR EACH TEAM
# n_steps = 100
# final_states_home = sparse.csr_matrix(np.identity(transitions.values.shape[0]))
# final_states_away = sparse.csr_matrix(np.identity(transitions.values.shape[0]))

# for i in range(n_steps):
#     final_states_home = final_states_home @ matchup_transitions_home[i]
#     final_states_away = final_states_away @ matchup_transitions_away[i]

# final_states_home = pd.DataFrame(data=final_states_home.toarray(), index=transitions.index, columns=transitions.columns)[absorbent_states]
# final_states_away = pd.DataFrame(data=final_states_away.toarray(), index=transitions.index, columns=transitions.columns)[absorbent_states]

# final_states_home.columns = [int(x.split('-')[-1]) for x in final_states_home.columns]
# final_states_away.columns = [int(x.split('-')[-1]) for x in final_states_away.columns]
# print(final_states_home)
# print(final_states_away)


# # MELT PIVOT AND RETAIN ONLY POSSIBLE STATES
# final_states_home = pd.melt(final_states_home, ignore_index=False, var_name='runs_scored', value_name='probability').reset_index()
# final_states_home = final_states_home.loc[final_states_home['probability'] > 0].copy()

# final_states_away = pd.melt(final_states_away, ignore_index=False, var_name='runs_scored', value_name='probability').reset_index()
# final_states_away = final_states_away.loc[final_states_away['probability'] > 0].copy()

# final_states_home[['inning', 'outs', 'runner_1b', 'runner_2b', 'runner_3b', 'home_score']] = np.array(final_states_home['game_state'].apply(lambda x: np.array(x.split('-'))).to_list()).astype(int)
# final_states_away[['inning', 'outs', 'runner_1b', 'runner_2b', 'runner_3b', 'away_score']] = np.array(final_states_away['game_state'].apply(lambda x: np.array(x.split('-'))).to_list()).astype(int)
# print(final_states_home)
# print(final_states_away)


# # SPLIT BATTING AND PITCHING STATES FOR EACH TEAM
# final_states_home_batting = final_states_home.copy()
# final_states_home_batting['half_inning'] = 'bottom'
# final_states_home_batting['effective_inning'] = final_states_home_batting['inning']

# final_states_away_batting = final_states_away.copy()
# final_states_away_batting['half_inning'] = 'top'
# final_states_away_batting['effective_inning'] = final_states_away_batting['inning']

# final_states_home_pitching = final_states_home.loc[final_states_home['outs'] + final_states_home['runner_1b'] + final_states_home['runner_2b'] + final_states_home['runner_3b'] == 0].copy()
# final_states_home_pitching['half_inning'] = 'top'
# final_states_home_pitching['effective_inning'] = final_states_home_pitching['inning']

# final_states_away_pitching = final_states_away.loc[final_states_away['outs'] + final_states_away['runner_1b'] + final_states_away['runner_2b'] + final_states_away['runner_3b'] == 0].copy()
# final_states_away_pitching['half_inning'] = 'bottom'
# final_states_away_pitching['effective_inning'] = final_states_away_pitching['inning'] - 1


# # JOIN STATES
# final_states_home_batting = final_states_home_batting.set_index('effective_inning')[['half_inning', 'inning', 'outs', 'runner_1b', 'runner_2b', 'runner_3b', 'home_score', 'runs_scored', 'probability']]
# final_states_home_pitching = final_states_home_pitching.set_index('effective_inning')[['home_score', 'runs_scored', 'probability']]

# final_states_away_batting = final_states_away_batting.set_index('effective_inning')[['half_inning', 'inning', 'outs', 'runner_1b', 'runner_2b', 'runner_3b', 'away_score', 'runs_scored', 'probability']]
# final_states_away_pitching = final_states_away_pitching.set_index('effective_inning')[['away_score', 'runs_scored', 'probability']]

# joined_top = final_states_home_pitching.join(final_states_away_batting, lsuffix='_home', rsuffix='_away').reset_index(drop=True)
# joined_bottom = final_states_home_batting.join(final_states_away_pitching, lsuffix='_home', rsuffix='_away').reset_index(drop=True)

# joined = pd.concat([joined_top, joined_bottom], axis=0)
# joined = joined.loc[joined['inning'] < 10].copy()
# print(joined)




### UNIVERSAL

# AGGREGATE PROBABILITY
joined['outcome'] = np.select([
    joined['runs_scored_home'] > joined['runs_scored_away'],
    joined['runs_scored_home'] < joined['runs_scored_away'],
    joined['runs_scored_home'] == joined['runs_scored_away'],
], ['home_win', 'away_win', 'tie'], None)
joined['joint_probability'] = joined['probability_home'] * joined['probability_away']

agg = joined.groupby(['half_inning', 'inning', 'outs', 'runner_1b', 'runner_2b', 'runner_3b', 'home_score', 'away_score', 'outcome'])['joint_probability'].sum().reset_index()
print(agg)


# PIVOT AND CLEAN
agg = pd.pivot_table(data=agg, index=['half_inning', 'inning', 'outs', 'runner_1b', 'runner_2b', 'runner_3b', 'home_score', 'away_score'], columns=['outcome'], values='joint_probability', fill_value=0)
agg = agg / agg.values.sum(axis=1, keepdims=True)
agg.fillna(0, inplace=True)
agg.reset_index(inplace=True)



transitions.to_csv('uninformed_transitions.csv')
agg.to_csv('uninformed_win_probability.csv', index=False)
