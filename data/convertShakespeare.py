#!/usr/bin/env python
# processes shakespeare code into the format our code accepts

fin = open('shakespeare.txt', 'r')
fout = open('new_speare.txt', 'w')

s = ""
fin.readline() # Get rid of first number, crimpin' my style
for line in fin.readlines():
    if line.strip().isdigit(): # new poem
        fout.write(s + '\n')
        s = ""
    elif line.strip() == "":
        continue
    else:
        s += line.strip().replace('.,?!', '').replace('-', ' ').lower() + ' | '
fout.write(s + '\n')
        
