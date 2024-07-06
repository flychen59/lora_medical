import os

string = ''

with open('/root/autodl-tmp/data_process/medical.test', 'r') as f:
    string = f.readlines()
    aa = 0

count = 0
for i in string:
    if i =='\n':
        count += 1

aa = 0