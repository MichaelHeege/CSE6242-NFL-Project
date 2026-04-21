import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy.stats import laplace_asymmetric
np.seterr(over='ignore') # We can tolerate the CDF calculation being 0.0

current_model = {'Coefficients': {
  'beta_down_2': np.float64(-0.08449692),
  'beta_down_3': np.float64(0.23197247),
  'beta_down_4': np.float64(-0.19837007),
  'beta_goal_to_go': np.float64(-3.08159529),
  'beta_ydstogo': np.float64(0.06538606),
  'beta_off_rank': np.float64(-0.01822458),
  'beta_def_rank': np.float64(0.02242741),
  'beta_pass_location_middle': np.float64(-0.06889333),
  'beta_pass_location_right': np.float64(-0.18591391),
  'beta_shotgun': np.float64(-0.76210986),
  'beta_air_yards': np.float64(-0.16885394),
  'intercept': np.float64(6.021534532838086)},
  'kappa': np.float64(0.24833478978976647),
  'loc': np.float64(-1.1478413708316806e-09),
  'scale': np.float64(1.3343713406042084)}

team_ranks_df = pd.read_csv(r'data\team_ranks.csv')

team_ranks_df.reset_index(drop=True, inplace=True)
team_list = list(set(team_ranks_df['team']))

# def create_model():
#   # Initialize an empty model
#   model = LinearRegression()
#
#   # Manually set the learned parameters
#   # Note: coef_ must be a 1D or 2D array depending on your target shape
#   model.coef_ = np.array([current_model['Coefficients']['beta_down_2'],
#                           current_model['Coefficients']['beta_down_3'],
#                           current_model['Coefficients']['beta_down_4'],
#                           current_model['Coefficients']['beta_goal_to_go'],
#                           current_model['Coefficients']['beta_ydstogo'],
#                           current_model['Coefficients']['beta_off_rank'],
#                           current_model['Coefficients']['beta_def_rank'],
#                           current_model['Coefficients']['beta_pass_location_middle'],
#                           current_model['Coefficients']['beta_pass_location_right'],
#                           current_model['Coefficients']['beta_shotgun'],
#                           current_model['Coefficients']['beta_air_yards']
#                           ])
#   model.intercept_ = current_model['Coefficients']['intercept']
#   return model

model = LinearRegression()
model.coef_ = np.array([current_model['Coefficients']['beta_down_2'],
                          current_model['Coefficients']['beta_down_3'],
                          current_model['Coefficients']['beta_down_4'],
                          current_model['Coefficients']['beta_goal_to_go'],
                          current_model['Coefficients']['beta_ydstogo'],
                          current_model['Coefficients']['beta_off_rank'],
                          current_model['Coefficients']['beta_def_rank'],
                          current_model['Coefficients']['beta_pass_location_middle'],
                          current_model['Coefficients']['beta_pass_location_right'],
                          current_model['Coefficients']['beta_shotgun'],
                          current_model['Coefficients']['beta_air_yards']
                          ])
model.intercept_ = current_model['Coefficients']['intercept']

def predict_yac_prob(lower, upper,
                     goal_to_go, down, ydstogo,
                     off_rank, def_rank, pass_location, shotgun,
                     air_yards, pass_completion):
  # Manage one-hot variables
  down_2 = down_3 = down_4 = 0
  match down:
    case '2':
      down_2 = 1
    case '3':
      down_3 = 1
    case '4':
      down_4 = 1
  pass_location_right = pass_location_middle = 0
  match pass_location:
    case 'middle':
      pass_location_middle = 1
    case 'right':
      pass_location_right = 1
  X_prediction = np.array([[
    down_2, down_3, down_4, goal_to_go, ydstogo,
    off_rank, def_rank, pass_location_middle, pass_location_right,
    shotgun, air_yards]])
  prediction = model.predict(X_prediction)

  # Fitted Kappa: 0.24833478978976647, Loc: -1.1478413708316806e-09, Scale: 1.3343713406042084
  # Parameters
  kappa = 0.24833478978976647
  loc = prediction
  scale = 1.3343713406042084

  cdf_upper = laplace_asymmetric.cdf(upper, kappa, loc, scale)
  cdf_lower = laplace_asymmetric.cdf(lower, kappa, loc, scale)

  prob = pass_completion * (cdf_upper - cdf_lower)
  # Add the uncompleted pass % to the (-0.5, 0.5)
  if lower == -0.5 and upper == 0.5:
      prob = prob + (1-pass_completion)

  return np.round(prob, decimals=4)


#############################################
#        Making it One Function
#############################################
def run_model(play):
  assert hasattr(play, 'LOS'), "play.LOS does not exist"
  assert 0 <= play.LOS <= 100, f"play.LOS must be between 0 and 100, got {play.LOS}"
  LOS = play.LOS

  assert hasattr(play, 'goal_to_go'), "play.goal_to_go does not exist"
  assert (play.goal_to_go in [0, 1]), f"play.goal_to_go must be 0 or 1, got {play.goal_to_go}"
  goal_to_go = play.goal_to_go

  assert hasattr(play, 'down'), "play.down does not exist"
  assert (str(play.down) in ['1', '2', '3', '4']), f"play.down must be 1, 2, 3, or 4, got {play.down}"
  down = str(play.down)

  assert hasattr(play, 'ydstogo'), "play.ydstogo does not exist"
  assert 0 <= play.ydstogo <= 100, f"play.ydstogo must be between 0 and 100, got {play.ydstogo}"
  ydstogo = play.ydstogo

  assert hasattr(play, 'posteam'), "play.posteam does not exist"
  assert (play.posteam in team_list), f"play.posteam not valid, got {play.posteam}"
  posteam = play.posteam

  assert hasattr(play, 'defteam'), "play.defteam does not exist"
  assert (play.defteam in team_list), f"play.defteam not valid, got {play.defteam}"
  defteam = play.defteam

  assert hasattr(play, 'season'), "play.season does not exist"
  assert 2010 <= play.season <= 2019, f"play.season must be between 2010 and 2019, got {play.season}"
  season = play.season

  assert hasattr(play, 'air_yards'), "play.air_yards does not exist"
  assert 0 <= play.air_yards <= 100, f"play.air_yards must be between 0 and 100, got {play.air_yards}"
  air_yards = play.air_yards

  assert hasattr(play, 'shotgun'), "play.shotgun does not exist"
  assert (play.shotgun in [0, 1]), f"play.shotgun must be 0 or 1, got {play.shotgun}"
  shotgun = play.shotgun

  assert hasattr(play, 'pass_location'), "play.pass_location does not exist"
  assert (play.pass_location in ['left', 'middle',
                                'right']), f"play.pass_location must be left, middle, or right, got {play.pass_location}"
  pass_location = play.pass_location

  assert hasattr(play, 'pass_completion'), "play.pass_completion does not exist"
  assert 0 <= play.pass_completion <= 1, f"play.pass_completion must be between 0 and 1, got {play.pass_completion}"
  pass_completion = play.pass_completion

  # Return 'off_rank' where 'Season' is 'season' AND 'team' is 'posteam''
  off_rank = team_ranks_df.loc[(team_ranks_df['Season'] == season) & (team_ranks_df['team'] == posteam), 'off_rank'].item()
  def_rank = team_ranks_df.loc[(team_ranks_df['Season'] == season) & (team_ranks_df['team'] == defteam), 'def_rank'].item()

  ranges = ([(-1e6, -LOS)]
            + [(-LOS, -LOS + (LOS % 5))]
            + [(-LOS + (LOS % 5) + 5 * (i - 1), -LOS + (LOS % 5) + 5 * i) for i in range(1, LOS // 5 + 1)]
            + [(-0.5, 0.5)]
            + [(5 * (i - 1), 5 * i) for i in range(1, (100 - LOS) // 5 + 1)]
            + [(95 + LOS % 5 - LOS, 100 - LOS)]
            + [(100 - LOS, 1e6)]
            )
  ranges = [(c[0], c[1] - 0.5) if c[1] == 0 else c for c in ranges]
  ranges = [(c[0] + 0.5, c[1]) if c[0] == 0 else c for c in ranges]
  ranges = [c for c in ranges if c[0] < c[1]]
  ranges = list(dict.fromkeys(ranges))

  output = pd.DataFrame()
  output['Gain/Loss Range'] = ranges
  output['Yardline Range'] = [(-10, 0)] + [(c[0] + LOS, c[1] + LOS) for c in ranges[1:-1]] + [(100, 110)]
  output['Label'] = ['Safety'] + [
    str(((['Loss of ', 'Gain of '][int(a > 0)] + str(abs(a)), str(abs(b)))))[1:-1].replace(', ', ' to ').replace(r"'",
                                                                                                                 "") for
    a, b in ranges[1:-1]] + ['Touchdown']
  output.loc[output['Gain/Loss Range'] == (-0.5, 0.5), 'Label'] = 'No Gain'
  output['Probabilities'] = output['Gain/Loss Range'].apply(lambda c: predict_yac_prob(
                                          lower=c[0],
                                          upper=c[1],
                                          goal_to_go=goal_to_go,
                                          down=down,
                                          ydstogo=ydstogo,
                                          off_rank=off_rank,
                                          def_rank=def_rank,
                                          pass_location=pass_location,
                                          shotgun=shotgun,
                                          air_yards=air_yards,
                                          pass_completion=pass_completion))
  first_down = predict_yac_prob(        lower=ydstogo,
                                        upper=1e6,
                                        goal_to_go=goal_to_go,
                                        down=down,
                                        ydstogo=ydstogo,
                                        off_rank=off_rank,
                                        def_rank=def_rank,
                                        pass_location=pass_location,
                                        shotgun=shotgun,
                                        air_yards=air_yards,
                                        pass_completion=pass_completion)

  return output.to_dict('list'), first_down

class Play:
    def __init__(self, LOS, goal_to_go, down, ydstogo, posteam, defteam, season, pass_location, shotgun, air_yards, pass_completion):
        self.LOS = LOS
        self.goal_to_go = goal_to_go
        self.down = down
        self.ydstogo = ydstogo
        self.posteam = posteam
        self.defteam = defteam
        self.season = season
        self.pass_location = pass_location
        self.shotgun = shotgun
        self.air_yards = air_yards
        self.pass_completion = pass_completion

# play1 = Play(48,0,'2',5,'SEA','DEN',2010,'right',0,5,0.50)
# results = run_model(play1)
# print(results)
