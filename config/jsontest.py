import json

with open("start.json", "r") as f:
    dict = json.load(f)
    print(dict['video'])