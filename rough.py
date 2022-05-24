from test import chatbot_response
import json
f = open("merged_dataset_intents.json")
data = json.load(f)
print(len(data['intents']))

# print(chatbot_response("Its a beautiful day"))
