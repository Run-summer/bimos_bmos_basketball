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

# tracking data (t_data) index
PLAYER_POSITIONS = slice(0, 20)

# event data index
EVENT_LABEL = 1
SCORE = 2
BALL_LOCATION = slice(3, 5)
BALL_PID = 5
CALC_FID = 6
LAST_CHOICE = 7
CALC_POS = slice(8, 10)
EVENT_LABELS = {
    'nonevent': 0,
    'pass': 1,
    'catch and pass': 2,
    'handoff catch and pass': 3,
    'catch': 4,
    'handoff pass': 5,
    'handoff catch and handoff pass': 6,
    'catch and handoff pass': 7,
    'handoff catch': 8,
    '2 point shot': 9,
    '3 point shot': 10,
    'turnover': 11
}
LAST_CHOICE_LABELS = {
    'pass': 0,    # pass-to-score sequence
    'dribble': 1  # dribble-to-score sequence
}

def load_t_data(game_number, onball=False):
    """
    Load SportVU tracking data for a specific game. 
    Set onball=True to obtain only on-ball tracking data.
    """
    game_str = str(game_number).zfill(3)
    if onball:
        t_data = loadmat(f"./basic_content/onball_scoreDataset/attackDataset_game{game_str}.mat")['data'][0]
    else:
        t_data = loadmat(f"./basic_content/modified_scoreDataset/attackDataset_game{game_str}.mat")['data'][0]
    return t_data

def load_event_data(game_number, onball=False):
    """
    Load event data for a specific game. 
    Set onball=True to obtain only on-ball event data.
    """
    if onball:
        events = loadmat("./onballevents_dataset.mat")['event'][0][game_number - 1][0]
    else:
        events = loadmat("./allevents_dataset.mat")['event'][0][game_number - 1][0]
    return events

def load_box_data():
    """
    Load box score statistics data.
    """
    box_data = pd.read_csv("basic_content/nba_datalength_updated.csv")
    return box_data

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

def make_transitionmodel_for_event(s_id, f_id, t_data, field_dimen = (14.,15.)):
    """
    Apply the transition model to the current pitch situation.
    """
    transitionmodel = np.array(pd.read_csv("groundwork/transitionmodel.csv", header=None))
    x_array = np.arange(0, field_dimen[0], 1) # [0, 1, ..., 13]
    y_array = np.arange(field_dimen[1]-1, -1, -1) # [14, 13, ..., 0]
    transition = np.empty((15, 14))
    dim_ball_id = get_pos_id([t_data[s_id][f_id][20], t_data[s_id][f_id][21]]) # [x, y] -> [iy, ix]
    center_id = [20, 20]

    for iy in y_array:
        for ix in x_array:
            transition[int(iy),int(ix)] = transitionmodel[int(center_id[0] + (iy - dim_ball_id[0])), int(center_id[1] + (ix - dim_ball_id[1]))]

    return transition

def making_likelihood_dataset(params, n_game, version, BMOS=True, last_choice=None):
    """
    Create result and expected arrays for likelihood estimation.
    Set BMOS=False to consider only PPCF/PBCF.
    Set last_choice="pass" or "dribble" to focus on pass-to-score or dribble-to-score sequences.
    """
    # obtain parameters used for function f(r,t) or f^b(t|r)
    fit_params, integral_xmin = res.get_params(params['player_accel'], params['att_reaction_time'], params['player_max_speed_att'])

    # load box score statistics
    box_data = load_box_data()

    # initialize output arrays
    result_array = []
    expected_BMOS = []
    s_ids = []

    count_shot = 0
    count_turnover = 0
    for game in tqdm(np.arange(1, n_game+1)):
        # load tracking, event, and box stats data per game
        t_data = load_t_data(game)
        events = load_event_data(game, onball=True)
        box_data_per_game = box_data.loc[box_data['game'] == game]

        for s_id, event in enumerate(events):
            shot_value = box_data_per_game['shot'].iloc[s_id] # 0 if turnover, 2 or 3 if a shot happens
            score_value = event[0][SCORE] # 0 if not a successful shot, else, 2 or 3
            count_value = shot_value # now set it to shot_value because we want to make 4000 shot and 600 turnover scenes
                                     # change it to score_value for making arrays based on score or not

            # check minimum data requirements, data is incomplete if len(event[0]) < 10
            if last_choice:
                condition_data = len(event[0]) == 10 and (event[0][LAST_CHOICE] == LAST_CHOICE_LABELS[last_choice])
            else:
                condition_data = len(event[0]) == 10

            if count_value == 0 and condition_data:
                count_turnover += 1
            elif count_value != 0 and condition_data:
                count_shot += 1
                    
            if count_value == 0 and count_turnover > 600:
                pass
                # if count_turnover == 600 + 1:
                #     print(f"turnover limit in game{game} s_id{s_id} with count {count_turnover}")
            elif count_value != 0 and count_shot > 4000:
                pass
                # if count_shot == 4000 + 1:
                #     print(f"shot limit in game{game} s_id{s_id} with count {count_shot}") 
            else:
                if condition_data: 
                    # obtain BMOS or BIMOS value
                    expected_BMOSa = calc_expected_obso(
                                        event, s_id, t_data, 
                                        params, fit_params, integral_xmin, version, BMOS=BMOS
                                        )
                    s_ids.append(s_id)
                    if BMOS:
                        if score_value != 0:
                            result_array.append(1)
                        else:
                            result_array.append(0)
                    else:
                        if shot_value != 0:
                            result_array.append(1)
                        else:
                            result_array.append(0)
                    expected_BMOS.append(expected_BMOSa)

    return result_array, expected_BMOS, s_ids

def calc_expected_obso(event, s_id, t_data, params, fit_params, integral_xmin, version, choose_fid=False, choose_location=False, BMOS=True):
    """
    Calculate BMOS or BIMOS value for a certain game sequence.
    Set choose_fid=[frame id where you want to calculate BMOS/BIMOS], else it is set to frame id when the ball was passed 
    in pass-to-score sequences, and when the ball possessor started to dribble in dribble-to-score sequences.
    Set choose_location=[x, y] if you want to set target position manually, else it is set to position where shot/turnover happened.
    """

    # load the score model
    score = np.array(pd.read_csv("groundwork/scoremodel.csv", header=None))
    
    if not choose_fid:
        calc_fid = event[0][CALC_FID]
    else:
        calc_fid = choose_fid

    if isinstance(choose_location, np.ndarray):
        cal_pos = choose_location
    elif choose_location is False:
        cal_pos = np.array(event[0][CALC_POS])
    
    cal_pos_id = get_pos_id(cal_pos) # [y, x]
    
    PPCFa, _ = scf.one_pixel_pitch_control_for_event(
                                cal_pos, s_id, int(calc_fid), t_data, 
                                params, fit_params, integral_xmin, version
                                )
    score_value = score[cal_pos_id[0], cal_pos_id[1]]
    if BMOS:
        if version == "BMOS":
            transition_plot = make_transitionmodel_for_event(s_id, int(calc_fid), t_data)
            transition_value = transition_plot[cal_pos_id[0], cal_pos_id[1]]
            expected_OBSOa = PPCFa * score_value * transition_value
        elif version == "BIMOS":
            expected_OBSOa = PPCFa * score_value
    else:
        if version == "BMOS":
            transition_plot = make_transitionmodel_for_event(s_id, int(calc_fid), t_data)
            transition_value = transition_plot[cal_pos_id[0], cal_pos_id[1]]
            expected_OBSOa = PPCFa * transition_value
        elif version == "BIMOS":
            expected_OBSOa = PPCFa
    
    return expected_OBSOa