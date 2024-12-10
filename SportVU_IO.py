"""
Module for reading and processing the dataset.

@author: Rikako Kono
"""

from scipy.io import loadmat
import numpy as np
import math
import pandas as pd
from tqdm import tqdm

import SportVU_ControlField as scf
import get_residual_param as res

# event data index
SCORE = 2
CALC_FID = 6
LAST_CHOICE = 7
CALC_POS = slice(8, 10)

LAST_CHOICE_LABELS = {
    'pass': 0,    # pass-to-score sequence
    'dribble': 1  # dribble-to-score sequence
}

def load_tracking_data(game_id, onball=False):
    """
    Load SportVU tracking data for a specific game. 
    Set onball=True to obtain only on-ball tracking data.
    """
    game_str = str(game_id).zfill(3)
    if onball:
        t_data = loadmat(f"./basic_content/onball_scoreDataset/attackDataset_game{game_str}.mat")['data'][0]
    else:
        t_data = loadmat(f"./basic_content/modified_scoreDataset/attackDataset_game{game_str}.mat")['data'][0]
    return t_data

def load_event_data(game_id, onball=False):
    """
    Load event data for a specific game. 
    Set onball=True to obtain only on-ball event data.
    """
    if onball:
        e_data = loadmat("./onballevents_dataset.mat")['event'][0][game_id - 1][0]
    else:
        e_data = loadmat("./allevents_dataset.mat")['event'][0][game_id - 1][0]
    return e_data

def load_box_data(game_id):
    """
    Load box score statistics data.
    """
    box_data = pd.read_csv("basic_content/nba_datalength_updated.csv")
    box_data_per_game = box_data.loc[box_data['game'] == game_id]
    return box_data_per_game

def get_pos_id(pos):
    """
    Convert pitch position to position array index.
    Input: [x, y], Output: [y_id, x_id].
    """
    FIELD_DIMS = (14, 15)
    yid, xid = 14 - math.floor(pos[1]), math.floor(pos[0])
    yid = max(0, min(yid, FIELD_DIMS[1] - 1))
    xid = max(0, min(xid, FIELD_DIMS[0] - 1))
    return [yid, xid]

def make_transitionmodel_for_event(t_data, s_id, f_id, field_dimen = (14.,15.)):
    """
    Apply the transition model to the current pitch situation.
    """
    transitionmodel = np.array(pd.read_csv("groundwork/transitionmodel.csv", header=None))
    x_array = np.arange(0, field_dimen[0], 1)     # [0, 1, ..., 13]
    y_array = np.arange(field_dimen[1]-1, -1, -1) # [14, 13, ..., 0]
    transition = np.empty((15, 14))
    dim_ball_id = get_pos_id([t_data[s_id][f_id][20], t_data[s_id][f_id][21]]) # [x, y] -> [iy, ix]
    center_id = [20, 20]

    for iy in y_array:
        for ix in x_array:
            transition[int(iy),int(ix)] = transitionmodel[int(center_id[0] + (iy - dim_ball_id[0])), int(center_id[1] + (ix - dim_ball_id[1]))]

    return transition

def making_likelihood_dataset(params, version, n_shot, n_turnover, last_choice=None):
    """
    Create result and expected arrays for likelihood estimation.
    Set last_choice="pass" or "dribble" to focus on pass-to-score or dribble-to-score sequences.
    """
    # obtain parameters used for function f(r,t) or f^b(t|r)
    fit_params, integral_xmin = res.get_params(params['player_accel'], 
                                               params['att_reaction_time'], 
                                               params['player_max_speed_att'])

    # initialize outputs and counters
    result_array = []
    expected_BIMOS = []
    s_ids = []
    count_shot = 0
    count_turnover = 0
    game_id = 1

    # Progress trackers
    def init_progress_bar(total, desc):
        return tqdm(total=total, desc=desc, position=0 if desc == "Shots Progress" else 1, leave=True)

    progress_shot = init_progress_bar(n_shot, "Shots Progress")
    progress_turnover = init_progress_bar(n_turnover, "Turnovers Progress")

    def meets_criteria(e_data_scene, last_choice):
        """
        Check if e_data_scene meets data requirements. 
        Data is incomplete if len(e_data_scene[0]) < 10
        """
        has_required_data = len(e_data_scene[0]) == 10
        if last_choice:
            return has_required_data and e_data_scene[0][LAST_CHOICE] == LAST_CHOICE_LABELS[last_choice]
        return has_required_data
    
    def update_counters(is_shot):
        """
        Update shot or turnover counters and progress bars.
        """
        nonlocal count_shot, count_turnover
        if is_shot:
            count_shot += 1
            progress_shot.update(1)
        else:
            count_turnover += 1
            progress_turnover.update(1)

    while (count_shot < n_shot or count_turnover < n_turnover) and game_id <= 630:
        # Load data for the current game
        t_data = load_tracking_data(game_id)
        e_data = load_event_data(game_id, onball=True)
        box_data_per_game = load_box_data(game_id)

        for s_id, e_data_scene in enumerate(e_data):
            if not meets_criteria(e_data_scene, last_choice):
                continue
            
            is_shot = box_data_per_game['shot'].iloc[s_id] > 0 
            is_score = e_data_scene[0][SCORE] > 0                     

            # Skip if we've reached the respective limits
            if (is_shot and count_shot >= n_shot) or (not is_shot and count_turnover >= n_turnover):
                continue
            update_counters(is_shot)
    
            # Calculate expected values
            expected_BIMOSa = calc_expected_values(
                                t_data, e_data_scene, s_id, 
                                params, fit_params, integral_xmin, version,
                                )
            expected_BIMOS.append(expected_BIMOSa)
            s_ids.append(s_id)

            # Append to result array based on version
            if version in {"BMOS", "BIMOS"}:
                result_array.append(1 if is_score else 0)
            else:
                result_array.append(1 if is_shot else 0)

        game_id += 1
    progress_shot.close()
    progress_turnover.close()
    return result_array, expected_BIMOS, s_ids

def calc_expected_values(t_data, e_data_scene, s_id, params, fit_params, integral_xmin, version, choose_fid=False, choose_location=False):
    """
    Calculate BMOS/PPCF or BIMOS/PPCF value for a certain game sequence.
    Set choose_fid=[frame id where you want to calculate BMOS/BIMOS], else it is set to frame id when the ball was passed 
    in pass-to-score sequences, and when the ball possessor started to dribble in dribble-to-score sequences.
    Set choose_location=[x, y] if you want to set target position manually, else it is set to position where shot/turnover happened.
    """

    # load the score model
    score = np.array(pd.read_csv("groundwork/scoremodel.csv", header=None))
    
    if not choose_fid:
        calc_fid = e_data_scene[0][CALC_FID]
    else:
        calc_fid = choose_fid

    if isinstance(choose_location, np.ndarray):
        cal_pos = choose_location
    elif choose_location is False:
        cal_pos = np.array(e_data_scene[0][CALC_POS])
    
    cal_pos_id = get_pos_id(cal_pos) # [y, x]
    
    PBCFa, _ = scf.one_pixel_pitch_control_for_event(
                        t_data, s_id, int(calc_fid), cal_pos,
                        params, fit_params, integral_xmin, version
                        )
    score_value = score[cal_pos_id[0], cal_pos_id[1]]
    if version == "BMOS":
        transition_plot = make_transitionmodel_for_event(t_data, s_id, int(calc_fid))
        transition_value = transition_plot[cal_pos_id[0], cal_pos_id[1]]
        expected_OBSOa = PBCFa * score_value * transition_value
    elif version == "BIMOS":
        expected_OBSOa = PBCFa * score_value
    elif version == "PPCF":
        transition_plot = make_transitionmodel_for_event(t_data, s_id, int(calc_fid))
        transition_value = transition_plot[cal_pos_id[0], cal_pos_id[1]]
        expected_OBSOa = PBCFa * transition_value
    elif version == "PBCF":
        expected_OBSOa = PBCFa
    
    return expected_OBSOa