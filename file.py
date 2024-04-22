import os
OUTPUT_FOLDER = "./result/GPT-4V"
ls = os.listdir(OUTPUT_FOLDER)
ls = [int(i) for i in ls]
ls.sort()
print(ls)