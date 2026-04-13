""" routes.py - Flask route definitions

Flask requires routes to be defined to know what data to provide for a given
URL. The routes provided are relative to the base hostname of the website, and
must begin with a slash."""


from flaskapp import app
from flask import request
from Interface.backend.play_handler import select_model, json_response, model_output, response


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

    #Get the play type (run or pass):
    play_type = request.args.get("play_type")

    #Select the model:
    selected_model = select_model(play_type)

    #Extract json from HTML: 
    data = json_response(play_type)

    #Get the model output
    result = model_output(selected_model, data)

    #Send response:
    return response(result)


