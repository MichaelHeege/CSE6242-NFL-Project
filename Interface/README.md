# Application Flow

**webpage -> request sent to Flask (Python) -> Python runs logic/model -> Flask returns template/data -> browser updates page**

---

## Project Structure

- **backend/**  
  This is where the model, processing functions, helper functions, and any data transformation code should live. Flask should import from this folder rather than storing model logic directly in `routes.py`.

- **flaskapp/**  
  Contains the Flask application code:
  - `routes.py` handles requests from the webpage and connects user input to backend/model logic
  - `templates/` contains HTML files rendered by Flask/Jinja
  - `static/` contains CSS, JavaScript, and other static files if the app needs them

- **run.py**  
  Used to start the Flask server locally.

---

## How It Works

1. A user opens `http://127.0.0.1:3001/`
2. Flask handles `GET /` in `flaskapp/routes.py`
3. Flask renders `flaskapp/templates/project_1APR2026.html`
4. The user interacts with the page and clicks the button to run the model
5. JavaScript sends a `POST` request to `/api/predict`
6. Flask reads the posted JSON, calls the backend/model logic, and returns JSON
7. JavaScript can use that JSON response to update the page

---

# Running the Application

### Run Flask App

#### Steps

1. Open a terminal.

2. Navigate to the `Interface/` folder containing `run.py`:

   ```bash
   cd path/to/project/Interface
   ```

3. If you are using Conda, activate the environment first.

4. Start the Flask server:

   ```bash
   python run.py
   ```

   On Windows, if Conda is active, prefer `python run.py` instead of `py run.py` because `py` may use a different interpreter.

5. Open your browser and go to:

   ```text
   http://127.0.0.1:3001/
   ```

6. The current app page rendered by Flask is:

   ```text
   flaskapp/templates/project_1APR2026.html
   ```

7. If you want to edit the current UI, this is the HTML file to update.

8. The model/data endpoint used by the page is:

   ```text
   /api/predict
   ```

   That route is defined in `flaskapp/routes.py`.

### Use a Different HTML File

1. Put the new HTML file inside:

   ```text
   Interface/flaskapp/templates/
   ```

   Example:

   ```text
   Interface/flaskapp/templates/my_new_page.html
   ```

2. Open `Interface/flaskapp/routes.py`.

3. Find the `GET /` route:

   ```python
   @app.route("/", methods=["GET"])
   def predictor():
       return render_template("project_1APR2026.html")
   ```

4. Change the template filename to the new file:

   ```python
   @app.route("/", methods=["GET"])
   def predictor():
       return render_template("my_new_page.html")
   ```

5. Save the file and restart Flask:

   ```bash
   python run.py
   ```

6. Refresh `http://127.0.0.1:3001/` in the browser.

### Notes on new HTML
Make sure new HTML file has the same JavaScript request as "project_1APR2026.html

```text
/api/predict
```

That means the new HTML should still send a `POST` request with JavaScript `fetch()` to `/api/predict` when the user clicks the button.


### Run Static HTML Only

Use this for standalone HTML files that do not need Flask or Python backend logic.

#### Steps

1. Start a local server:

   **Windows**

   ```bash
   py -m http.server 8000
   ```

   **Mac/Linux**

   ```bash
   python3 -m http.server 8000
   ```

2. Open your browser and go to:

   ```text
   http://localhost:8000
   ```

3. Navigate to the desired HTML file such as `ProjectDemo.html`.

---

# Notes

- Use Flask when the page requires **Python logic, data processing, or models**
- Use `http.server` for **static HTML only**
- If dependencies are missing, install them with:

  ```bash
  python -m pip install -r requirements.txt
  ```

- Use `CTRL+C` in the terminal to stop the server

# Flask Notes

- Flask uses Jinja as its templating engine
- `{{ }}` is a Jinja placeholder for dynamic content
- `{% %}` is a Jinja directive
- `{# #}` is a Jinja comment
- Rendering is the process of turning a template into a complete HTML page

# Flask Functions

- `render_template()` loads an HTML template and returns the rendered page
- In this app, `GET /` renders `project_1APR2026.html`
- `POST /api/predict` receives input data and returns a JSON response

# Route Example

```python
@app.route("/", methods=["GET"])
def predictor():
    return render_template("project_1APR2026.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    ...
```

# HTML Notes

- An HTML `form` is a container used to collect user input
- Common input elements include text fields, dropdowns, checkboxes, radio buttons, and submit buttons
- In the current app, JavaScript `fetch()` is also used to send data to Flask without reloading the page

# Query String Example

If a page uses `GET`, browser controls can appear in the URL as query parameters.

Example:

```text
/?play_type=run
```
