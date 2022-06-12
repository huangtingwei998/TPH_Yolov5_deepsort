import json

payload = {
    'system': 'C:/Users/huangtingwei/Desktop/飞行器pythonProject17/'
}
with open('system.json','w') as f:
    json.dump(payload, f)

with open("system.json", "r") as f:
    dict = json.load(f)
    print(dict['system'])

import json

data = {
    'flag': False,
    'ID': 0
}
with open('findID.json', 'w+') as f:
    json.dump(data, f)


with open("findID.json", "r") as f:
    dict = json.load(f)
    print(dict['ID'])
    print(dict['flag'])