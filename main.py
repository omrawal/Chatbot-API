from flask import Flask, json, jsonify
from test import chatbot_response
app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World! this is chatbot api</p>"


@app.route("/<string:s>")
def called(s):
    result = {
        "user_query": s,
        "chatbot_response": chatbot_response(s)
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(host="localhost", port=5000, debug=True)
