""" routes.py - Flask route definitions

Flask requires routes to be defined to know what data to provide for a given
URL. The routes provided are relative to the base hostname of the website, and
must begin with a slash."""


from flaskapp import app
from flask import render_template, request
from Interface.backend.play_handler import Play, get_model


# The following two lines define two routes for the Flask app, one for just
# '/', which is the default route for a host, and one for '/index', which is
# a common name for the main page of a site.
#
# Both of these routes provide the exact same data - that is, whatever is
# produced by calling `index()` below


@app.route('/', methods=["GET", "POST"])


#When request arrives for /index, call the index() function 
@app.route('/index', methods=["GET", "POST"])
def predict():

    """Renders the html template"""
    #Gets the play data from the request (HTML)and creates a Play object with it
    play = Play(
        game_clock=request.args.get("game_clock"),
        quarter=request.args.get("quarter"),
        down=request.args.get("down"),
        play_type=request.args.get("play_type"),
        formation=request.args.get("formation")
    )

    #Gets the model based on the play type
    model = get_model(play.play_type)

    #Gets the output of the model. model() returns the play function for the model 
    model_output = model(play)

    # NEED TO EDIT BELOW CODE #

    # Renders the template using Jinja
    # !!!!!!!!! Need to determine how outputs look like and how to pass them to the template !!!!!!!!!

    return render_template(
        'index.html',
        table=table,
        header=header,
        username=username(),
        option_list=dropdown_options,
        filter_class=filter_class,


    )
