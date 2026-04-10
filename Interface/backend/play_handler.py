"""
This file contains functions for retrieving data and calling specific models
to process the data
"""

class Play:
    def __init__(self, game_clock, quarter, down, play_type, formation):
        self.game_clock = game_clock
        self.quarter = quarter
        self.down = down
        self.play_type = play_type
        self.formation = formation


def get_model(play_type):
    if play_type == "run":
        # NEED TO EDIT BELOW CODE #
        # Place holder for the run model, which should be called when the play type is "run"
        # Will return the play function for the run model, which should be called in the predict() function in routes.py
        return run_model 
    
    elif play_type == "pass":
        # NEED TO EDIT BELOW CODE #
        # Place holder for the pass model, which should be called when the play type is "pass"
        # Will return the play function for the pass model, which should be called in the predict() function in routes.py
        return pass_model
    
    else:
        return "place holder model"
    

