"""
Module for visualizing the dataset.

@author: Rikako Kono
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import matplotlib.animation as animation
from tqdm import tqdm
import datetime

import SportVU_IO as sio
import SportVU_ControlField as scf

# Constants for t_data
PLAYER_POSITIONS = slice(0, 20)
BALL_POSITION = slice(20, 23)
PLAYER_VELOCITIES = slice(23, 43)
PLAYER_IDS = slice(48, 58)
JERSEY_NUMBER = slice(58, 68)
BALL_PID_IDX = 69

# event data index
CALC_FID = 6
CALC_POS = slice(8, 10)

# Constants for eventdata
EVENT_LABEL = 1
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

def plotCourt():
    court_path = './basic_content/nba_court_T.png'
    img = mpimg.imread(court_path)
    plt.imshow(img, extent=[0, 28, 0, 15], zorder=0)
    plt.xlim(0, 14)
    plt.ylim(0, 15)

def plot_pitchcontrol_for_frame(
        game_id, s_id, f_id, params, fit_params, integral_xmin, version, 
        include_player_velocities=True, annotate=False, BID=True, target_position=False, 
        colorbar=True, axis=True, title=True, field_dimen = (14.,15.), colormap="Reds"):
    """
    Plots heatmaps for a specific frame.

    Parameters:
    -----------
        game_id, s_id, f_id: IDs for the game, scene, and frame.
        params: Parameters for pitch control calculation.
        fit_params: Parameters for tau_true - tau_exp distribution fitting.
        integral_xmin: Integration limit for pitch control.
        version: One of "BMOS", "BIMOS", "PPCF", or "PBCF".
        include_player_velocities: If True, displays player velocity arrows.
        annotate: If True, displays player IDs on the plot.
        BID: If True, highlights the player in ball possession.
        target_position: If True, shows the final target position.
        colorbar: If True, includes a colorbar in the plot.
        axis: If True, shows axis ticks and labels.
        title: If True, adds a title to the plot.
    -----------
    """

    # Data loading
    t_data = sio.load_tracking_data(game_id)
    t_data_frame = t_data[s_id][f_id]
    e_data_scene = sio.load_event_data(game_id)[s_id]
    attPitch, _, _, _ = scf.generate_pitch_control_for_event(
                                t_data, s_id, f_id, params, fit_params, integral_xmin, version)

    # Calculate expected values
    if version in ["BMOS", "BIMOS"]:
        score = np.array(pd.read_csv("groundwork/scoremodel.csv", header=None))
        attValue = attPitch * score
        if version == "BMOS":
            transition_plot = sio.make_transitionmodel_for_event(s_id, f_id, t_data)
            attValue *= transition_plot
    elif version in ["PPCF", "PBCF"]:
        attValue = attPitch
        if version == "PPCF":
            transition_plot = sio.make_transitionmodel_for_event(s_id, f_id, t_data)
            attValue *= transition_plot

    # Extracting player and ball positions
    dim_att = [ t_data_frame[0:10:2], t_data_frame[1:10:2] ]
    dim_def = [ t_data_frame[10:20:2], t_data_frame[11:20:2] ]
    dim_ball = t_data_frame[BALL_POSITION][:2]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 5) if axis else (4.8, 4.8))
    if axis:
        fig.subplots_adjust(left=0.01, bottom=0.08, right=0.99, top=0.95)
        plt.text(15.2, -0.9, '[m]', ha='center')
    else:
        fig.subplots_adjust(left=0.01, bottom=0.08, right=0.99, top=0.95)
        plt.xticks([])
        plt.yticks([])  
        
    ax.imshow(attValue, cmap=colormap, extent=(0, field_dimen[0], 0, field_dimen[1]), alpha=0.9)
    if colorbar:
        plt.colorbar(ax.imshow(attValue, cmap=colormap, alpha=0.9), ax=ax)

    plotCourt()

    ax.scatter(*dim_att, s=100, edgecolor ='r', c = "white")
    ax.scatter(*dim_def, s=100, edgecolor='b', c = "white")
    ax.scatter(*dim_ball, s=30, c = "black")

    # Plot target position
    if target_position:
        try:
            calc_position = e_data_scene[0][CALC_POS]
        except:
            calc_position = [7., 7.5]
            print(f"error happens in game: {game_id} scene: {s_id}")
        ax.scatter(*calc_position, marker='x', s=20, color='black')

    # Highlight ball possessor
    if BID and t_data_frame[BALL_PID_IDX] > 0:
        bid_idx = int(t_data_frame[BALL_PID_IDX] - 1)
        ax.scatter(
            t_data_frame[PLAYER_POSITIONS][2 * bid_idx], 
            t_data_frame[PLAYER_POSITIONS][2 * bid_idx + 1], 
            s=17, facecolors='none', edgecolors='black'
            )

    # Plot player velocity arrows
    if include_player_velocities:
        for i in range(10):
            plt.quiver(
                t_data_frame[PLAYER_POSITIONS][i*2], 
                t_data_frame[PLAYER_POSITIONS][i*2+1], 
                t_data_frame[PLAYER_VELOCITIES][i*2], 
                t_data_frame[PLAYER_VELOCITIES][i*2+1], 
                angles='xy', scale_units='xy', scale=1, color='black'
                )

    # Add player id numbers
    if annotate:
        for i in range(10):
            player_id = int(t_data_frame[PLAYER_IDS][i])
            plt.text(
                t_data_frame[PLAYER_POSITIONS][i*2], 
                t_data_frame[PLAYER_POSITIONS][i*2+1], 
                f'{player_id}', fontsize=8
                )
    
    # Add title
    if title:
        plt.title(f'Game {game_id} - Event {s_id} - Frame {f_id}')
        
    plt.show()

def plot_pitchcontrol_for_sequence(
        game_id, s_id, params, fit_params, integral_xmin, version, heatmap=True, 
        EVENT=True, JERSEY=True, BID=False, axis=False, title=True, field_dimen=(14., 15.)):
    """
    Plots animation for a specific scene.

    Parameters:
    -----------
        game_id, s_id, f_id: IDs for the game, scene, and frame.
        params: Parameters for pitch control calculation.
        fit_params: Parameters for tau_true - tau_exp distribution fitting.
        integral_xmin: Integration limit for pitch control.
        version: One of "BMOS", "BIMOS", "PPCF", or "PBCF".
        heatmap: If True, displays heatmap.
        EVENT: If True, displays event labels.
        JERSEY: If Ture, displays jersey numbers
        BID: If True, highlights the player in ball possession.
        axis: If True, shows axis ticks and labels.
        title: If True, adds a title to the plot.
    -----------
    """

    def get_key_from_value(d, val):
        keys = [k for k, v in d.items() if v == val]
        return keys[0]
    
    def extract_date_info(gamename):
        parts = gamename.split('_')

        day = int(parts[1])
        month = int(parts[0])
        year = int(parts[2])
        
        suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        day_formatted = f"{day}{suffix}"
        
        month_name = datetime.date(1900, month, 1).strftime('%b')
        
        return day_formatted, month_name, str(year)

    # Data loading
    t_data = sio.load_tracking_data(game_id)
    e_data_scene = sio.load_event_data(game_id)[s_id]
    box_data_per_game = sio.load_box_data(game_id)

    # Obtain date info
    gamename = box_data_per_game['gamename'].iloc[s_id]
    date, month, year = extract_date_info(gamename)

    # Obtain team info
    team_id_O = box_data_per_game['team_O'].iloc[s_id]
    team_id_D = box_data_per_game['team_D'].iloc[s_id]
    team_name_O = sio.load_team_name(team_id_O)
    team_name_D = sio.load_team_name(team_id_D)
    
    if heatmap and version in ["BMOS", "BIMOS"]:
        score = np.array(pd.read_csv("groundwork/scoremodel.csv", header=None))

    calc_fid, calc_position = 0, [7., 7.5]
    try:
        calc_fid = e_data_scene[0][CALC_FID]
        calc_position = e_data_scene[0][CALC_POS]
    except KeyError as e:
        print(f"Proper event data does not exist in game: {game_id}, scene: {s_id}: {e}")

    # Precalculate frame data
    precalculated_data = []
    for f_id, t_data_frame in enumerate(tqdm(t_data[s_id])):
        frame_info = {
            'dim_att': [ t_data_frame[0:10:2],  t_data_frame[1:10:2]],
            'dim_def': [ t_data_frame[10:20:2],  t_data_frame[11:20:2]],
            'dim_ball':  t_data_frame[20:22]
        }

        if heatmap:
            attPitch, *_ = scf.generate_pitch_control_for_event(
                                            t_data, s_id, f_id, params, fit_params, 
                                            integral_xmin, version)                                           
            if version in ["BMOS", "BIMOS"]:
                frame_info['attValue'] = attPitch * score
                if version == "BMOS":
                    transition_plot = sio.make_transitionmodel_for_event(s_id, f_id, t_data)
                    frame_info['attValue'] *= transition_plot
            elif version in ["PPCF", "PBCF"]:
                frame_info['attValue'] = attPitch
                if version == "PPCF":
                    transition_plot = sio.make_transitionmodel_for_event(s_id, f_id, t_data)
                    frame_info['attValue'] *= transition_plot

        if EVENT:
            frame_info['event_label'] = e_data_scene[f_id][EVENT_LABEL]

        if JERSEY:
            frame_info['jersey_number'] = t_data_frame[JERSEY_NUMBER]

        precalculated_data.append(frame_info)

    # Animation
    fig, ax = plt.subplots()

    def animate(f_id):
        ax.clear()
        t_data_frame = t_data[s_id][f_id]
        frame_info = precalculated_data[f_id]

        # Plot heatmap
        if heatmap:
            ax.imshow(frame_info['attValue'], cmap='Reds', vmin=0., vmax=1., 
                    extent=(0, field_dimen[0], 0, field_dimen[1]), alpha=0.9)
            
        # Plot players and ball
        ax.scatter(*frame_info['dim_att'],  s=100, edgecolor ='r', c = "white")
        ax.scatter(*frame_info['dim_def'], s=100, edgecolor='b', c = "white")
        ax.scatter(*frame_info['dim_ball'], s=30, c = "black")
        ax.scatter(*calc_position, marker='x', s=20, color='black')

        # Highlight ball possessor
        if BID:
            bid_idx = int(t_data_frame[BALL_PID_IDX] - 1)
            ax.scatter(
                t_data_frame[PLAYER_POSITIONS][2 * bid_idx], 
                t_data_frame[PLAYER_POSITIONS][2 * bid_idx + 1], 
                s=60, facecolors='none', edgecolors='black'
            )

        # Add event labels
        if EVENT:
            ax.text(*frame_info['dim_ball'], f"{frame_info['event_label']}", fontsize=8)

        # Add jersey numbers
        if JERSEY:
            for i in np.arange(10):
                jersey_numbers = int(frame_info["jersey_number"][i])
                if i < 5:
                    x, y = frame_info['dim_att'][0][i], frame_info['dim_att'][1][i]
                else:
                    x, y = frame_info['dim_def'][0][i-5], frame_info['dim_def'][1][i-5]
                ax.text(x, y, 
                        f'{jersey_numbers}', 
                        fontsize=8, horizontalalignment='center', verticalalignment='center',)

        # Add titles
        fid_str = str(f_id).zfill(3)
        if not title:
            key = get_key_from_value(EVENT_LABELS, e_data_scene[len(e_data_scene)-1][EVENT_LABEL])
            if f_id == calc_fid:
                ax.set_title(f'Game {game_id} - Event {s_id} - Frame {fid_str} - {key}', color='red')
            else:
                ax.set_title(f'Game {game_id} - Event {s_id} - Frame {fid_str} - {key}')
        else:
            ax.set_title('')
            ax.text(0.08, 1.025, team_name_O, color='red', fontsize=12, ha='center', transform=ax.transAxes)
            ax.text(0.17, 1.025, 'vs.', color='black', fontsize=12, ha='center', transform=ax.transAxes)
            ax.text(0.25, 1.025, team_name_D, color='blue', fontsize=12, ha='center', transform=ax.transAxes)
            ax.text(0.65, 1.025, date + ' ' + month + '. ' +  year + ' - Frame ' + fid_str, 
                    color='black', fontsize=12, ha='center', transform=ax.transAxes)

        if not axis:
            plt.xticks([])
            plt.yticks([])  

        plotCourt()

    ani = animation.FuncAnimation(fig, animate, frames=len(t_data[s_id]), interval=100)
    plt.show()