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

#####################################
#   Import models 
#####################################

# Use package-relative imports because play_handler.py is loaded as part of
# the backend package (from backend.play_handler import ...).
from .Run_Model import run_model                         #Michaels model
from .YAC_Model import run_model as run_yac_model        #Mikes model 
from .Pass_Model_JP import predict_pass_probability      #JP model

#####################################
#   Helper Functions for Play Class and Model Input Processing
#####################################

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


#####################################
#   Play Class (Defines the inputs for the models NEEDED IN UI) 
#####################################

#Play class to hold play data:
class Play:
    def __init__(
        self,
        # ----------------------------
        # Shared inputs used by multiple models
        # ----------------------------
        LOS=None,
        goal_to_go=None,
        down=None,
        ydstogo=None,
        shotgun=None,
        air_yards=None,
        pass_location=None,

        # ----------------------------
        # Run model inputs: Michael
        # ----------------------------
        # LOS=None,
        # goal_to_go=None,
        # down=None,
        # ydstogo=None,
        formation=None,
        run_location=None,
        run_gap=None,

        # ----------------------------
        # Pass completion model inputs: JP
        # Shared inputs are commented out so the UI person can see what is new.
        # ----------------------------
        # air_yards=None,
        yardline_100=None,
        # ydstogo=None,
        # down=None,
        qtr=None,
        # shotgun=None,
        off_rank=None,
        def_rank=None,
        # pass_location=None,

        # ----------------------------
        # YAC model inputs: Mike
        # Put new/non-duplicate fields first for this section.
        # ----------------------------
        posteam=None,
        defteam=None,
        season=None,
        pass_completion=None,  # Comes from the pass model output.
        # LOS=None,
        # goal_to_go=None,
        # down=None,
        # ydstogo=None,
        # pass_location=None,
        # shotgun=None,
        # air_yards=None,
    ):
        # Shared inputs
        self.LOS = LOS
        self.goal_to_go = goal_to_go
        self.down = down
        self.ydstogo = ydstogo
        self.shotgun = shotgun
        self.air_yards = air_yards
        self.pass_location = pass_location

        # Run model fields
        self.formation = formation
        self.run_location = run_location
        self.run_gap = run_gap

        # Pass model fields
        self.yardline_100 = yardline_100
        self.qtr = qtr
        self.off_rank = off_rank
        self.def_rank = def_rank

        # YAC model fields
        self.posteam = posteam
        self.defteam = defteam
        self.season = season
        self.pass_completion = pass_completion

        # Helpful aliases for model/UI code that may use alternate names.
        self.offense_team = posteam
        self.defense_team = defteam
        self.quarter = qtr
        self.pass_attempt_length = air_yards


#####################################
#   Extracts model
#####################################

def select_model(play_type):
    if play_type == "run":
        #return the run model
        return run_model
    elif play_type == "pass":
        #return a list of pass models
        return [predict_pass_probability, run_yac_model]
    else:
        raise ValueError("Invalid play type. Please select either 'run' or 'pass'.")
    
#####################################
#   Extracts json from HTML: 
#####################################

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
            shotgun=_int(shotgun, 0),
            formation=formation,
            run_location=(request_data.get("run_location") or request_data.get("runDirection")),
            run_gap=request_data.get("run_gap")
        )

        return run_play


    elif play_type == "pass":
        # Extracts the pass play data from the request (HTML) and creates a Play object
        pass_play = Play(
            LOS=_int(request_data.get("LOS")),
            goal_to_go=_int(request_data.get("goal_to_go")),
            down=_int(request_data.get("down")),
            ydstogo=_int(request_data.get("ydstogo")),
            shotgun=_int(request_data.get("shotgun"), 0),
            air_yards=_int(request_data.get("air_yards")),
            pass_location=request_data.get("pass_location"),
            yardline_100=_int(request_data.get("yardline_100")),
            qtr=_int(request_data.get("qtr")),
            off_rank=_int(request_data.get("off_rank")),
            def_rank=_int(request_data.get("def_rank")),
            posteam=request_data.get("posteam"),
            defteam=request_data.get("defteam"),
            season=_int(request_data.get("season")),
            pass_completion=_float(request_data.get("pass_completion"))
        )

        return pass_play


#####################################
#  Runs the model: (Output = python dictionary)
#####################################

def model_output(selected_model, data):
    #If the selected model is a list 
    if isinstance(selected_model, list):
        #Run the pass model and get the output:
        pass_model_output = selected_model[0](data)

        #Plug the pass model output into the YAC model and get the output: 
        data.pass_completion = pass_model_output
        yac_model_output = selected_model[1](data)

        #Return the output of the YAC model to the HTML page:
        output = {
            "pass_completion": pass_model_output,
            "yac_output": yac_model_output
        }
    else: 
        output = selected_model(data)

    return output

#####################################
#   Sends response in JSON to HTML: 
#####################################
def response(model_output):
    res = make_response(jsonify(model_output), 200)
    return res
