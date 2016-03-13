#!/usr/bin/env python
from HMM import *

print("\nSingle Shakespeare")
s, l = parseFile('data/singlespeare.txt')
fromt, tot, ll = todict(s, l)
print("Num tokens: " + str(len(fromt)))
sspeare = HMM(len(fromt), fromt, tot, k=5)
sspeare.learn(ll, tol=0.1, prt=True)
lstwords = sspeare.gettops()
for l in lstwords:
    print(l)

print("\nSmall Shakespeare")
s, l = parseFile('data/smallspeare.txt')
fromt, tot, ll = todict(s, l)
print("Num tokens: " + str(len(fromt)))
smallspeare = HMM(len(fromt), fromt, tot, k=5)
smallspeare.learn(ll, tol=0.1, prt=True)
lstwords = smallspeare.gettops()
for l in lstwords:
    print(l)

print("\nFull Shakespeare")
s, l = parseFile('data/new_speare.txt')
fromt, tot, ll = todict(s, l)
print("Num tokens: " + str(len(fromt)))
full = HMM(len(fromt), fromt, tot, k=5)
full.learn(ll, tol=0.1, prt=True)
lstwords = full.gettops(numWords=10)
for l in lstwords:
    print(l)
