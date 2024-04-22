import os
import statistics
import json

folderPath = input("Your result folder? (eg. result/CogVLM): ")
iters = []

for sub in os.listdir(folderPath):
    jsonFilePath = sub + '.json'
    if sub == '.ipynb_checkpoints':
        continue
        
    sub = os.path.join(folderPath, sub)
    jsonFilePath = os.path.join(sub, jsonFilePath)
    
    if os.path.exists(jsonFilePath):
        with open(jsonFilePath, 'r') as jf:
            data = jf.read()
            data = json.loads(data)
            iter = data['iterations']
            iters.append(iter)
    else: print(sub)

print("Len:  ", len(os.listdir(folderPath)))
print("Mean iter: ", statistics.mean(iters))
print("Std iter:  ", statistics.stdev(iters))