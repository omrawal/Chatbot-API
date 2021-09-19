# search flask minimal app
#https://youtu.be/Jzv3G5iDLvw
# password hyok

from flask import Flask, json,jsonify
from chatbot_fun import chat
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/<string:s>")
def called(s):
    result={
        "user_query":s,
        "chatbot_response":chat(s)
    }
    return jsonify(result)












if __name__=='__main__':
    app.run(debug=True)