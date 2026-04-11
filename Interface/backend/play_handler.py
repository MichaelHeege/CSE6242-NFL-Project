"""
This file contains functions for retrieving data and calling specific models
to process the data
"""

from flask import request

class Play:
    def __init__(
        self,
        game_clock=None,
        quarter=None,
        down=None,
        play_type=None,
        formation=None,
        ydstogo=None,
        shotgun=None,
        qb_dropback=None,
        goal_to_go=None,
        gap_x_location=None,
        LOS=None
    ):
        self.game_clock = game_clock
        self.quarter = quarter
        self.down = down
        self.play_type = play_type
        self.formation = formation
        self.ydstogo = ydstogo
        self.shotgun = shotgun
        self.qb_dropback = qb_dropback
        self.goal_to_go = goal_to_go
        self.gap_x_location = gap_x_location
        self.LOS = LOS


def model_output(play_type):
    if play_type == "run":
        # NEED TO EDIT BELOW CODE #
        # Place holder for the run model, which should be called when the play type is "run"
        # Will return the play function for the run model, which should be called in the predict() function in routes.py
        
        #Gets the play data from the request (HTML)and creates a Play object with it
        run_play = Play(
            game_clock=request.args.get("game_clock"),
            quarter=request.args.get("quarter"),
            down=request.args.get("down"),
            play_type=request.args.get("play_type"),
            formation=request.args.get("formation")
        )
        
        Run_play_output = run_model(run_play) # Calls the run model with the play data and gets the output
            
        return Run_play_output 
    
    elif play_type == "pass":
        # NEED TO EDIT BELOW CODE #
        # Place holder for the pass model, which should be called when the play type is "pass"
        # Will return the play function for the pass model, which should be called in the predict() function in routes.py
    
        #Gets the play data from the request (HTML)and creates a Play object with it
        pass_play = Play(
            game_clock=request.args.get("game_clock"),
            quarter=request.args.get("quarter"),
            down=request.args.get("down"),
            play_type=request.args.get("play_type"),
            formation=request.args.get("formation")
        )
        
        Pass_play_output = pass_model(pass_play) # Calls the pass model with the play data and gets the output
            
        return Pass_play_output 
    
    else:
        return "Error: Invalid play type. Please select either 'run' or 'pass'."
    

