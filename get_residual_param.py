"""
Module for calculating the function f(r,t) or f^b(t|r), which is the probability that a player will reach target position r before t.

Event Data Structure:
all_events = sio.load_event_data(game, onball=False): event data per game
    ---all_events[s_id][f_id]: event data per frame, where s_id is a scene id, and f_id is a frame id
        --- [f_id, event_label, score, ball_x, ball_y, ball_holder_pid_idx, calc_fid, last_choice, calc_posx, calc_posy]
            event_label: current frame's on-ball event name (see EVENT_LABELS dictionary)
            ball_x, ball_y: current ball position
            ball_holder_pid_idx: player index who holds the ball (1-10), 0 if no one holds the ball
            calc_fid: frame id when BMOS/BIMOS value is calculated
            last_choice: pass-to-score or dribble-to-score
            calc_posx, calc_posy: target position where BMOS/BIMOS value is calculated

You can obtain on_ball_events by setting onball=True.

@author: Rikako Kono
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import skewcauchy
from scipy.optimize import curve_fit
import seaborn as sns

import SportVU_IO as sio

# event data index
EVENT_LABEL = 1
BALL_PID_IDX = 5
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

# tracking data (t_data) index
PLAYER_POSITIONS = slice(0, 20)
PLAYER_VELOCITIES = slice(23, 43)
SHOT_CLOCK_ID = 47

# Constant
N_TEST_GAMES = 50 # use the first 50 games to shorten the processing time

def get_params(accel, r_time, vmax):
    """
    Plot and fit tau_true - tau_exp distribution with skewcauchy.cdf function.
    This distribution depends on player acceleration, reaction time, and maximum velocity.
    Return resulting params and integration xmin used in get_cdf_value() function.
    """
    hist_array = []

    # obtain tau_true - tau_exp array as hist_array
    for game in tqdm(np.arange(1, N_TEST_GAMES+1)):
        # load trackiing data and on-ball event data
        t_data = sio.load_t_data(game, onball=True)
        events = sio.load_event_data(game, onball=True)

        for s_data, event in zip(t_data, events):
            if len(event) >= 2:
                # when more than two on-ball events happen
                for f_data, f_data_next, frame, frame_next in zip(s_data[:-1], s_data[1:], event[:-1], event[1:]):
                    if frame[EVENT_LABEL] in [EVENT_LABELS['pass'], EVENT_LABELS['catch and pass'], EVENT_LABELS['catch'], EVENT_LABELS['handoff catch']]:
                        true_tau = f_data[SHOT_CLOCK_ID] - f_data_next[SHOT_CLOCK_ID]
                        exp_tau = calc_expected_tau(f_data, f_data_next, frame, frame_next, accel, r_time, vmax)
                        if true_tau > 0:
                            hist_array.append(true_tau - exp_tau)
    try:
        # plot and fit tau_true - tau_exp distribution
        hist = plt.hist(hist_array, bins=int((np.max(hist_array)-np.min(hist_array))/0.4), range=None, density=True, cumulative=True)
        x = [(hist[1][i] + hist[1][i+1]) / 2 for i in np.arange(len(hist[1])-1)]
        y = hist[0]

        initial_parameter = [0.5, 0, 1]
        fit_params, _ = curve_fit(skewcauchy.cdf, x, y, p0=initial_parameter)

        # comment out when you want to draw a rigid graph
        plt.ioff()

        # uncomment when you want to draw a rigid graph
        """
        # Plot Fit Lines
        sns.set()
        sns.set_style('darkgrid')
        sns.set_palette('gray')
        np.set_printoptions(suppress=True)
        hist = plt.hist(hist_array, bins=int((np.max(hist_array)-np.min(hist_array))/0.2), range=None, density=True, cumulative=False)
        x = [(hist[1][i]+hist[1][i+1])/2 for i in np.arange(len(hist[1])-1)]
        y = hist[0]

        initial_parameter = [0.5, 0, 1]

        fit_params, _ = curve_fit(skewcauchy.pdf, x, y, p0=initial_parameter)

        fig, ax = plt.subplots(1, 1)
        plt.scatter(x, y, s=15)
        x_fit = np.linspace(-3, 7, 1000)
        a_fit, loc_fit, scale_fit = fit_params
        ax.plot(x_fit, 
                skewcauchy.pdf(x_fit, a_fit, loc_fit, scale_fit),
                'black', label='skewcauchy pdf')
        plt.xlim(-3, 7)
        plt.xlabel(r"$\tau_{true}-\tau_{exp}$ [s]")
        plt.ylabel("probability density")
        # plt.title("Asymmetric Lorentizian Fit")
        plt.show()
        """
        return fit_params, np.min(hist[1])
    except ValueError:
        # when fitting does not go well
        print("fitting did not go well")
        return [0], 100

def calc_time_to_intercept(r_start, r_final, v_current, accel, r_time, vmax):
    """
    Calculate time taken for a player to reach target position r_final.
    """
    vini = np.linalg.norm(v_current)

    # calc position after reaction time
    adjust_position = r_start + v_current * r_time

    # calc time to reach target position from adjust_position, given vmax isn't set
    t = (-vini / accel + np.sqrt(vini ** 2 / accel ** 2 + 2 * np.linalg.norm(r_final - adjust_position) / accel))
    
    # consider the situation velocity exceeds vmax
    if vini + t * accel > vmax:
        limit_time = (vmax - vini) / accel
        remaining_distance = np.linalg.norm(r_final - adjust_position) - (vini * limit_time + 0.5 * accel * limit_time ** 2) 
        time_to_intercept = r_time + limit_time + remaining_distance / vmax
    else:
        time_to_intercept = r_time + t
        
    return time_to_intercept

def calc_expected_tau(f_data, f_data_next, frame, frame_next, accel, r_time, vmax):
    """
    Return calc_time_to_intercept() depending on pass or dribble situation.
    """
    if frame[EVENT_LABEL] in [EVENT_LABELS['pass'], EVENT_LABELS['catch and pass']]:
        idx = int(frame_next[BALL_PID_IDX]) - 1
    elif frame[EVENT_LABEL] in [EVENT_LABELS['catch'], EVENT_LABELS['handoff catch']]:
        idx = int(frame[BALL_PID_IDX]) - 1

    r_start = f_data[PLAYER_POSITIONS][idx*2:idx*2+2]
    r_final = f_data_next[PLAYER_POSITIONS][idx*2:idx*2+2]
    v_current = f_data[PLAYER_VELOCITIES][idx*2:idx*2+2]

    return calc_time_to_intercept(r_start, r_final, v_current, accel, r_time, vmax)

def skewed_cauchy_distribution(x, a, loc, scale):
    bottom_bottom = scale**2 * (1 + a * np.sign(x - loc))**2
    bottom_top = (x - loc)**2
    bottom = scale * np.pi * (1 + bottom_top/bottom_bottom)
    return 1 / bottom

def get_cdf_value(x, fit_params, integral_xmin, num_points=1000):
    """
    Return propability that a player reaches target position before the ball, T,
    by integrating tau_true - tau_exp distribution from integral_xmin to T-time_to_intercept.
    """
    a, loc, scale = fit_params[0], fit_params[1], fit_params[2]
    if x <= integral_xmin:
        return 0
    else:
        x_values = np.linspace(integral_xmin, x, num_points)
        y_values = skewed_cauchy_distribution(x_values, a, loc, scale)
        return np.trapz(y_values, x_values)