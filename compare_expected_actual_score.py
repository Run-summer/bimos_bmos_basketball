"""
Code for comparing expected values with actual ones.

@author: Rikako Kono
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns; sns.set()
sns.set_style('darkgrid')
sns.set_palette('gray')
np.set_printoptions(suppress=True)

import SportVU_IO  as sio
import SportVU_ControlField as mpc 
import get_residual_param as scf

# Constants
GOAL_LOCATION = np.array([1.575, 7.5]) # [m]
version = "BIMOS"

# Constants for index
LAST_CHOICE = 7
CALC_POS = slice(8, 10)

LAST_CHOICE_LABELS = {
    'pass': 0,
    'dribble': 1
}

# Parameter info
params = mpc.default_model_params(accel=7.758029732151973, 
                                  kappa=1.0220227143365117, 
                                  lam=36.60277280686902, 
                                  att_reaction_time=0.1566666417841336, 
                                  def_reaction_time=0.4954688947311042)
fit_params, integral_xmin = scf.get_params(params['player_accel'], 
                                           params['att_reaction_time'], 
                                           params['player_max_speed_att'])

results = {score_type: {'actual': [], 'expected': []} 
           for score_type in ['total', 'pass', 'dribble']}

for game in tqdm(np.arange(581, 631)):
    # Load game data
    t_data = sio.load_tracking_data(game)
    e_data = sio.load_event_data(game)
    box_data_per_game = sio.load_box_data(game)

    # Team identification
    team_ids = [box_data_per_game['team_O'].iloc[0], box_data_per_game['team_D'].iloc[0]]

    # Initialize score trackers
    game_scores = {team_id: {'total': 0, 'pass': 0, 'dribble': 0} for team_id in team_ids}
    expected_scores = {team_id: {'total': 0, 'pass': 0, 'dribble': 0} for team_id in team_ids}

    for s_id, e_data_scene in enumerate(e_data):
        if len(e_data_scene[0]) != 10:
            continue

        # Extract scene details
        last_choice = e_data_scene[0][LAST_CHOICE]
        offense_team_id = box_data_per_game['team_O'].iloc[s_id]
        shot_attempt_location = box_data_per_game['shot'].iloc[s_id]
        actual_score = box_data_per_game['score3'].iloc[s_id]

        # Expected score calculation
        expected_score = sio.calc_expected_values(t_data, e_data_scene, s_id, params, 
                                                  fit_params, integral_xmin, version)
        if shot_attempt_location == 0:
            calc_pos = np.array(e_data_scene[0][CALC_POS])
            distance_to_goal = np.linalg.norm(GOAL_LOCATION - calc_pos)
            expected_score *= 3. if distance_to_goal >= 6.5 else 2.
        else:
            expected_score *= shot_attempt_location

        # Determine choice type
        choice_type = 'pass' if last_choice == LAST_CHOICE_LABELS['pass'] else 'dribble'

        # Update scores
        game_scores[offense_team_id]['total'] += actual_score
        game_scores[offense_team_id][choice_type] += actual_score
        expected_scores[offense_team_id]['total'] += expected_score
        expected_scores[offense_team_id][choice_type] += expected_score

    for team_id in team_ids:
        for score_type in ['total', 'pass', 'dribble']:
            if game_scores[team_id][score_type] > 0:
                results[score_type]['actual'].append(game_scores[team_id][score_type])
                results[score_type]['expected'].append(expected_scores[team_id][score_type])

# Compute R² metrics
r2_metrics = {key: r2_score(results[key]['actual'], results[key]['expected']) for key in results}
print(f"R² Metrics: {r2_metrics}")

# Visualization
fig = plt.figure(figsize=(20, 4))
fig.subplots_adjust(left=0.04, bottom=0.15, right=0.99, top=0.93)
color_map = {'total': 'black', 'pass': 'blue', 'dribble': 'green'}

for i, score_type in enumerate(['total', 'pass', 'dribble'], 1):
    plt.subplot(1, 3, i)
    plt.scatter(results[score_type]['actual'], results[score_type]['expected'], 
                c=color_map[score_type], s=5)
    plt.plot(plt.xlim(), plt.xlim(), 'red', label='y=x')
    plt.title(f"{score_type.capitalize()} Events")
    plt.xlabel("Actual Score per game")
    plt.ylabel("Expected Score per game")
    plt.text(plt.xlim()[1]*0.2, plt.ylim()[1]*0.8, 
             f"R² ({score_type.capitalize()}): {r2_metrics[score_type]:.2f}", 
             color='red')

plt.show()