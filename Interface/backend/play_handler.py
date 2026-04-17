"""
This file contains functions for retrieving data and calling specific models
to process the data
"""

import sys
from pathlib import Path

from flask import request, jsonify, make_response

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Run_Model import run_model

# Helper functions for parsing input values
def _int(val, default=None):
    try:
        if val is None:
            return default
        return int(val)
    except (ValueError, TypeError):
        return default

def _float(val, default=None):
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

#Play class to hold play data:
class Play:
    def __init__(
        self,
        #Run play parameters:
        LOS=None,
        goal_to_go=None,
        down=None,
        ydstogo=None,
        qb_dropback=None,
        shotgun=None,
        run_location =None,
        run_gap=None,

        # Pass play parameters (Dummy parameters for now, will be updated when pass model is implemented):
        game_clock=None,
        quarter=None,
        play_type=None,
        formation=None,

    ):
        #Run play parameters:
        self.LOS = LOS
        self.goal_to_go = goal_to_go
        self.down = down
        self.ydstogo = ydstogo
        self.qb_dropback = qb_dropback
        self.shotgun = shotgun
        self.run_location  = run_location
        self.run_gap = run_gap

        #Pass play parameters (Dummy parameters for now, will be updated when pass model is implemented): 
        self.game_clock = game_clock
        self.quarter = quarter
        self.play_type = play_type
        self.formation = formation


#Extracts model: 
def select_model(play_type):
    if play_type == "run":
        return run_model
    elif play_type == "pass":
        # Placeholder for pass model function, which should be returned when the play type is "pass"
        return None
    else:
        raise ValueError("Invalid play type. Please select either 'run' or 'pass'.")
    

#Extracts json from HTML: 
def json_response(play_type):

    #Gets the JSON data from the HTML request
    request_data = request.get_json(silent=True) or {}

    if play_type == "run":
        down = request_data.get("down")
        if isinstance(down, str) and down.endswith(("st", "nd", "rd", "th")):
            down = down[0]

        formation = request_data.get("formation")
        shotgun = request_data.get("shotgun")
        if shotgun is None and formation is not None:
            shotgun = 1 if str(formation).lower() == "shotgun" else 0

        # Extracts the run play data from the request (HTML) and creates a Play object
        run_play = Play(
            LOS=_int(request_data.get("LOS")),
            goal_to_go=_int(request_data.get("goal_to_go")),
            down=_int(down),
            ydstogo=_int(request_data.get("ydstogo")),
            qb_dropback=_int(request_data.get("qb_dropback"), 0),
            shotgun=_int(shotgun, 0),
            run_location=(request_data.get("run_location") or request_data.get("runDirection")),
            run_gap=request_data.get("run_gap")
        )

        return run_play


    elif play_type == "pass":
        # Extracts the pass play data from the request (HTML) and creates a Play object
        pass

#Runs the model: (Output = python dictionary)
def model_output(selected_model, data):
    return selected_model(data)

#Sends response in JSON to HTML: 
def response(model_output):
    res = make_response(jsonify(model_output), 200)
    return res
