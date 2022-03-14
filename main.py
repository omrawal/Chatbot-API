from flask import Flask, json, jsonify, request
from test import chatbot_response
import ast
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


@app.route('/', methods=['GET', 'POST'])
def returnChatbotResponse():
    if request.method == 'POST':
        # userQuery = ast.literal_eval(request.args.get('userQuery'))
        userQuery = (request.args.get('userQuery'))
        print('############')
        print(userQuery)
        print('############')
        print('type= ', type(userQuery))
        result = {
            "user_query": userQuery,
            "chatbot_response": chatbot_response(userQuery.lower())
        }
        return jsonify(result)
    else:
        return "<p>Please use proper API POST request call for chatbot response</p>"


if __name__ == '__main__':
    app.run(host="localhost", port=5000, debug=True)
