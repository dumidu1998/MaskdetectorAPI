import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Create some test data for our catalog in the form of a list of dictionaries.

books = {'Sucess':'false'}



@app.route('/', methods=['GET'])
def home():
    f = open("mytxt.txt", "r")
    out = f.read()
    if out == "Detected":
        books = {'Sucess':'true'}
    else:
        books = {'Sucess':'false'}
    return jsonify(books)

app.run()