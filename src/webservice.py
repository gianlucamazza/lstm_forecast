import os
from flask import Flask, send_from_directory, render_template_string

app = Flask(__name__, static_folder='../docs')

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plotly Graphs</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }
        header {
            background-color: #4CAF50;
            color: white;
            padding: 1em 0;
            text-align: center;
        }
        main {
            padding: 1em;
        }
        ul {
            list-style-type: none;
            padding: 0;
        }
        li {
            background: white;
            margin: 0.5em 0;
            padding: 1em;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        li a {
            text-decoration: none;
            color: #333;
        }
        li a:hover {
            color: #4CAF50;
        }
        .search-bar {
            margin-bottom: 1em;
        }
        .search-bar input {
            width: 100%;
            padding: 0.5em;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
    </style>
    <script>
        function searchGraphs() {
            let input = document.getElementById('searchInput').value.toLowerCase();
            let ul = document.getElementById('graphList');
            let li = ul.getElementsByTagName('li');
            for (let i = 0; i < li.length; i++) {
                let a = li[i].getElementsByTagName('a')[0];
                if (a.innerHTML.toLowerCase().indexOf(input) > -1) {
                    li[i].style.display = "";
                } else {
                    li[i].style.display = "none";
                }
            }
        }
    </script>
</head>
<body>
    <header>
        <h1>Available Plotly Graphs</h1>
    </header>
    <main>
        <div class="search-bar">
            <input type="text" id="searchInput" onkeyup="searchGraphs()" placeholder="Search for graphs...">
        </div>
        <ul id="graphList">
            {% for file in files %}
            <li><a href="{{ file }}" target="_blank">{{ file }}</a></li>
            {% endfor %}
        </ul>
    </main>
</body>
</html>
"""

@app.route('/')
def index():
    files = [f for f in os.listdir(app.static_folder) if f.endswith('_predictions.html')]
    return render_template_string(INDEX_TEMPLATE, files=files)

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
