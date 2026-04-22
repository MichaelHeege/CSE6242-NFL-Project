"""routes.py - Flask route definitions."""


from flaskapp import app
from flask import jsonify, render_template, request
from backend.play_handler import select_model, json_response, model_output, response

#Route = URL path + type of request (GET, POST, etc.) it accepts

#GET = Give me something (e.g., a webpage, data, etc.)
#Purpose: Shows the page
@app.route("/", methods=["GET"])
def predictor():
    return render_template("project_1APR2026.html")


#POST = Here is some data, do something with it (e.g., run a model, save it to a database, etc.)
#When user sends data to /api/predict, this function will be called
#Purpose: Process the users input. Data is posted to /api/predict
@app.route("/api/predict", methods=["POST"])
def predict():
    play_type = (request.args.get("play_type") or (request.get_json(silent=True) or {}).get("play_type"))

    if play_type is None:
        return jsonify({"error": "Missing play_type. Please select either 'run' or 'pass'."}), 400

    #Selects model based on play type
    selected_model = select_model(play_type)

    #Extracts data from HTML and creates a Play object 
    data = json_response(play_type)

    #Runs the model and gets the output
    result = model_output(selected_model, data)

    #Returns the result as a JSON response to the HTML page
    return response(result)
