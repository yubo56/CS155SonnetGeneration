#!/usr/bin/env python
# generate transition/emission matricies for the various datasets
# single shakespeare, few shakespeare, truncated shakespeare and full
# full runs too slowly.

from HMM import *
import pickle

f = open('sspeare10.pkl', 'wb')
print("\nSingle Shakespeare")
s, l = parseFile('data/singlespeare.txt')
fromt, tot, ll = todict(s, l)
print("Num tokens: " + str(len(fromt)))
sspeare = HMM(len(fromt), fromt, tot, k=10)
sspeare.learn(ll, tol=0.1, prt=True)
lstwords = sspeare.gettops(numWords=10)
for l in lstwords:
    print(l)
pickle.dump(sspeare, f, -1)

f = open('small10.pkl', 'wb')
print("\nSmall Shakespeare")
s, l = parseFile('data/smallspeare.txt')
fromt, tot, ll = todict(s, l)
print("Num tokens: " + str(len(fromt)))
smallspeare = HMM(len(fromt), fromt, tot, k=10)
smallspeare.learn(ll, tol=0.03, prt=True)
lstwords = smallspeare.gettops(numWords=10)
for l in lstwords:
    print(l)
pickle.dump(smallspeare, f, -1)

f = open('full.pkl', 'wb')
print("\nFull Shakespeare")
s, l = parseFile('data/short_speare.txt')
fromt, tot, ll = todict(s, l)
print("Num tokens: " + str(len(fromt)))
full = HMM(len(fromt), fromt, tot, k=5)
full.learn(ll, tol=0.05, prt=True)
lstwords = full.gettops(numWords=10)
for l in lstwords:
    print(l)
pickle.dump(full, f, -1)

f = open('full10.pkl', 'wb')
print("\nFull Shakespeare")
s, l = parseFile('data/short_speare.txt')
fromt, tot, ll = todict(s, l)
print("Num tokens: " + str(len(fromt)))
full = HMM(len(fromt), fromt, tot, k=10)
full.learn(ll, tol=0.05, prt=True)
lstwords = full.gettops(numWords=10)
for l in lstwords:
    print(l)
pickle.dump(full, f, -1)

f = open('realfull.pkl', 'wb')
print("\nFull Shakespeare")
s, l = parseFile('data/new_speare.txt')
fromt, tot, ll = todict(s, l)
print("Num tokens: " + str(len(fromt)))
full = HMM(len(fromt), fromt, tot, k=5)
full.learn(ll, tol=0.1, prt=True)
lstwords = full.gettops(numWords=10)
for l in lstwords:
    print(l)
pickle.dump(full, f, -1)
