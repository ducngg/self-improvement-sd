import json
import glob
REF_FOLDER = './json/json-13b'
dataset = {}
i=0
for filename in glob.iglob(f'{REF_FOLDER}/*.json'):
    image_id = int(filename[16:-5])
    with open(filename, 'r') as f:
        file = json.load(f)
    dataset[image_id] = file['input']
    i+=1
print(i)
with open('dataset.json', 'w') as f:
    json.dump(dataset, f)