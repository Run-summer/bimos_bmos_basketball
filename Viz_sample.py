"""
Sample code for visualizing the dataset.

@author: Rikako Kono
"""

import SportVU_ControlField as scf 
import SportVU_Viz as sviz
import SportVU_IO as sio
import get_residual_param as res

# Constants for event data index
SCORE = 2
CALC_FID = 6

game, s_id = 1, 9
version = "BIMOS"
t_data = sio.load_tracking_data(game)
e_data = sio.load_event_data(game)
# f_id = 50
f_id = int(e_data[s_id][0][CALC_FID]) # Frame id where BIMOS is calculated; 
                                      # when a player passed in pass-to-score and 
                                      # when a player started dribbling in dribble-to-score

# Parameter info
accel, kappa, lam, att_reaction_time, def_reaction_time = 7.758029732151973, 1.0220227143365117, 36.60277280686902, 0.1566666417841336, 0.4954688947311042 
params = scf.default_model_params(accel=accel, kappa=kappa, lam=lam, 
                                  att_reaction_time=att_reaction_time, 
                                  def_reaction_time=def_reaction_time)

fit_params, integral_xmin = res.get_params(params['player_accel'], 
                                           params['att_reaction_time'], 
                                           params['player_max_speed_att'])

""" 
Plot heatmap for the specific game frame 
"""
# sviz.plot_pitchcontrol_for_frame(game, s_id, f_id, params, fit_params, integral_xmin, version, 
#                                  include_player_velocities=True, annotate=False, BID=False, 
#                                  target_position=True, colorbar=True, axis=True, title=True)

"""
Plot animation for the specific game sence
"""
sviz.plot_pitchcontrol_for_sequence(game, s_id, params, fit_params, integral_xmin, version, 
                                    heatmap=True, EVENT=True, JERSEY=True, BID=False, axis=True, title=True)