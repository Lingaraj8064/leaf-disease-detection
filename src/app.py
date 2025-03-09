from flask import Flask, render_template, request
import os

app = Flask(__name__, template_folder="../templates")  # Set correct template path

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")  # Make sure this file exists

if __name__ == "__main__":
    app.run(debug=True)
