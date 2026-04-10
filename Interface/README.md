

# Application Flow

**webpage → request sent to Flask (Python) → Python runs logic/model → Flask renders HTML → browser updates page**

---

## Project Structure

- **backend/**  
  This is where the model, processing functions, helper functions, and any data transformation code should live. Flask should import from this folder rather than storing model logic directly in `routes.py`.

- **data/**  
  Contains the datasets used by the application. This may include raw input data, cleaned data, intermediate outputs, or files generated for the model. This folder should only store data files.

- **flaskapp/**  
  Contains the Flask application code:
  - `routes.py` handles requests from the webpage and connects user input to backend/model logic  
  - `templates/` contains dynamic HTML files rendered using Flask/Jinja  
  - `static/` contains CSS, JavaScript, and other static files used for frontend behavior  

- **run.py**  
  Used to start the Flask server locally

---

## How It Works

1. A user interacts with the webpage (e.g., selects items from a dropdown)
2. The browser sends a request to the Flask application  
3. Flask reads the user input and calls the appropriate model or processing functions  
4. The model processes the data and generates results  
5. Flask passes the results into an HTML template  
6. The browser displays the updated page  


<br> 

<br/>

# Running the Application

---

### 🔹 Run Flask App (Dynamic HTML)

#### Steps:

1. Open a terminal

2. Navigate to the directory containing `run.py`:

   ```bash
   cd path/to/project
   ```

3. Start the Flask server:

   ```bash
   py run.py
   ```

   *(or `python run.py` depending on your setup)*

4. Open your browser and go to:

   ```text
   http://127.0.0.1:3001/
   ```

5. This will load `index.html` from:

   ```
   flaskapp/templates/
   ```

---

### 🔹 Run Static HTML (No Flask)

Use this for standalone HTML files (no backend logic).

#### Steps:

1. Start a local server:

   **Windows:**

   ```bash
   py -m http.server 8000
   ```

   **Mac/Linux:**

   ```bash
   python3 -m http.server 8000
   ```

2. Open your browser and go to:

   ```text
   http://localhost:8000
   ```

3. Navigate to the desired HTML file (e.g., `ProjectDemo.html`)

---
<br> 

<br/>

#  Notes

* Use Flask when the page requires **Python logic, data processing, or models**
* Use `http.server` for **static HTML only**
* If dependencies are missing, install them with:

  ```bash
  py -m pip install -r requirements.txt
  ```
* Use  CTRL+C in terminal to end servers



# Flask Notes
* FLASK uses Jinja as its templating engine. The HTML is considered a Jinja template (Since it has placeholders/directives/Jinja comments)
* {{ }} is a Jinja placeholder for dynamic content (For example the model outputs)
* The operation that converts a template into a complete HTML page is called rendering

* "{% }" = Jinja directive
* "{# }" = Jinja comments


#Flask functions
* To render a template can import a function render_template() 
* This function takes a HTML template filename and a variable list of template arguments and returns the same template but with all the placeholder in it replaced with actual values 


# HTML notes

* HTML "form" is used to collect user input. It is a container for different types of input elements such as: text fields, checkboxes, radio buttons, submit buttons etc.

```html
<form>
  <span>Show species class:</span>
  <select name="class" onchange="this.form.submit()">
    <option value="">All</option>
    {% for o in option_list %}
      <option value="{{ o }}" {% if filter_class == o %} selected {% endif %}>{{ o }}</option>
    {% endfor %}
  </select>
</form>
```
# Breakdown of code

```html
<select name="class">
```
- Gives the control the key "class"

```html
<option>
```
- Provides a value (if absent, the option text is used)
- The browser reads the selected option's value

```html
onchange="this.form.submit()"
```
- Calls the browser to submit the form immediately when the user picks an option

- The form has no `method`, so the browser uses GET (method just tells how to send the form data )
- The browser encodes all named controls as a query string: ?
- With no `action`, the browser submits to the current page URL. (action just tells which URL to send the form to)

Example:
```
/index?class=Mammal
```




