import os

for DIR in os.listdir('result'):
    print(DIR, len(os.listdir('result/'+DIR)))