import pandas as pd
import numpy as np
from scipy import stats

#############################################
#                INPUTS
#############################################
down = '2'
ydstogo = 5
shotgun = 0
gap_x_location = 'middle' #Options: 'left_end', 'left_guard', 'left_tackle', 'middle', 'right_tackle', 'right_guard', 'right_end'
qb_dropback = 0
goal_to_go   = 0

LOS = 38

##############################################
#             MY CODE
##############################################

# Model definition
current_model = {'dist': stats.nct,
 'dist_name': 'Skew-t',
 'beta': np.array([-0.02961292,  0.0219228 ,  2.23608327,  0.34037474,  0.01526812,
        -0.10960522, -0.06916322,  1.52349615,  1.05502495,  1.37977475,
         0.22001933,  1.50491111,  1.14769691, -2.12226287, -0.43434344,
         0.02256054,  0.03148844,  0.06140939,  0.06559673,  0.04407096,
        -0.14679915, -0.41020082, -0.27269458, -0.39051472, -0.02799404,
        -0.43215745, -0.29824158,  0.96365088]),
 'beta_dict': {'loc_goal_to_go': np.float64(-0.0296129219488656),
  'loc_ydstogo': np.float64(0.02192279571812747),
  'loc_qb_dropback': np.float64(2.236083273143995),
  'loc_shotgun': np.float64(0.3403747447385068),
  'loc_down_2': np.float64(0.01526811845912526),
  'loc_down_3': np.float64(-0.10960521814829113),
  'loc_down_4': np.float64(-0.06916321613802051),
  'loc_gap_x_location_left_guard': np.float64(1.5234961483750513),
  'loc_gap_x_location_left_tackle': np.float64(1.0550249463054207),
  'loc_gap_x_location_middle': np.float64(1.3797747471611808),
  'loc_gap_x_location_right_end': np.float64(0.22001933198472234),
  'loc_gap_x_location_right_guard': np.float64(1.5049111063934488),
  'loc_gap_x_location_right_tackle': np.float64(1.1476969123223904),
  'loc_intercept': np.float64(-2.1222628724265102),
  'log_scale_goal_to_go': np.float64(-0.43434343576365997),
  'log_scale_ydstogo': np.float64(0.022560543207444536),
  'log_scale_qb_dropback': np.float64(0.03148843785811818),
  'log_scale_shotgun': np.float64(0.06140938764525669),
  'log_scale_down_2': np.float64(0.06559673004926889),
  'log_scale_down_3': np.float64(0.04407096358242609),
  'log_scale_down_4': np.float64(-0.14679914590507068),
  'log_scale_gap_x_location_left_guard': np.float64(-0.41020081568604455),
  'log_scale_gap_x_location_left_tackle': np.float64(-0.27269458091926885),
  'log_scale_gap_x_location_middle': np.float64(-0.39051471737781895),
  'log_scale_gap_x_location_right_end': np.float64(-0.02799404461129181),
  'log_scale_gap_x_location_right_guard': np.float64(-0.43215745073795464),
  'log_scale_gap_x_location_right_tackle': np.float64(-0.2982415757528241),
  'log_scale_intercept': np.float64(0.9636508769792489)},
 'shape_params': (np.float64(2.501175221282317),
  np.float64(1.3726078051141841)),
 'n_shape': 2,
 'n_features': 13,
 'features': ['goal_to_go',
  'down',
  'ydstogo',
  'qb_dropback',
  'shotgun',
  'gap_x_location'],
 'yards_col': 'yards_gained'}

# Helper
def _predict_params(model, X):
    beta       = model["beta"]
    n_features = model["n_features"]
    shapes     = model["shape_params"]

    loc   = X @ beta[:n_features] + beta[n_features]
    scale = np.exp(X @ beta[n_features+1:2*n_features+1] + beta[2*n_features+1])

    params_array = [tuple(shapes) + (loc[i], scale[i]) for i in range(len(X))]
    return params_array, loc
    
# Probability function
def predict_run_success_prob(model, lower, upper,
                             goal_to_go, down, ydstogo,
                             qb_dropback, shotgun,
                             gap_x_location):
    dist     = model["dist"]
    features = model["features"]

    input_dict = {
        "goal_to_go":     goal_to_go,
        "down":           down,
        "ydstogo":        ydstogo,
        "qb_dropback":    qb_dropback,
        "shotgun":        shotgun,
        "gap_x_location": gap_x_location,
    }

    train_cols = [
        k.replace("loc_", "", 1)
        for k in model["beta_dict"]
        if k.startswith("loc_") and k != "loc_intercept"
    ]

    x = np.zeros(len(train_cols))
    for j, col in enumerate(train_cols):
        if col in features:
            input_val = input_dict.get(col)
            if input_val is not None:
                x[j] = float(input_val)
            continue
    
        for feature in features:
            prefix = feature + "_"
            if col.startswith(prefix):
                val = col[len(prefix):]
                input_val = input_dict.get(feature)
                if input_val is None:
                    continue
                try:
                    if float(val) == float(input_val):
                        x[j] = 1.0
                except (ValueError, TypeError):
                    if str(val) == str(input_val):
                        x[j] = 1.0
                break


    X = x.reshape(1, -1)
    params_array, _ = _predict_params(model, X)
    p = params_array[0]

    prob = dist.cdf(upper, *p) - dist.cdf(lower, *p)
    return float(prob)


#############################################
#            Getting Ranges
#############################################
ranges = ([(-1e6,-LOS)]
          +[(-LOS,-LOS+(LOS%5))]
          +[(-LOS+(LOS%5)+5*(i-1),-LOS+(LOS%5)+5*i) for i in range(1,LOS//5+1)]
          +[(-0.5,0.5)]
          +[(5*(i-1),5*i) for i in range(1,(100-LOS)//5+1)]
          +[(95+LOS%5-LOS,100-LOS)]
          +[(100-LOS,1e6)]
         )
ranges = [(c[0], c[1] - 0.5) if c[1] == 0 else c for c in ranges]
ranges = [(c[0] + 0.5, c[1]) if c[0] == 0 else c for c in ranges]
ranges = [c for c in ranges if c[0]<c[1]]
ranges = list(dict.fromkeys(ranges))





# Final output
output = pd.DataFrame()
output['Gain/Loss Range'] = ranges
output['Yardline Range'] = [(-10,0)]+[(c[0]+LOS,c[1]+LOS) for c in ranges[1:-1]]+[(100,110)]
output['Label'] = ['Safety']+[str(((['Loss of ','Gain of '][int(a>0)]+str(abs(a)),str(abs(b)))))[1:-1].replace(', ', ' to ').replace(r"'","") for a,b in ranges[1:-1]]+['Touchdown']
output.loc[output['Gain/Loss Range']==(-0.5,0.5),'Label'] = 'No Gain'
output['Probabilities'] = output['Gain/Loss Range'].apply(lambda c: predict_run_success_prob(
                                                                                model      = current_model,
                                                                                lower      = c[0],
                                                                                upper      = c[1],
                                                                                goal_to_go = 0,
                                                                                down       = down,
                                                                                ydstogo    = ydstogo,
                                                                                qb_dropback= qb_dropback,
                                                                                shotgun    = shotgun,
                                                                                gap_x_location    = gap_x_location)
                                                         )
print(sum(output['Probabilities']))
display(output)