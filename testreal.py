#!/usr/bin/env python
from HMM import *

print("\nSingle Shakespeare")
s, l = parseFile('data/singlespeare.txt')
fromt, tot, ll = todict(s, l)
sspeare = HMM(len(fromt), fromt, tot, k=5)
sspeare.learn(ll, tol=0.1, prt=True)
lstwords = sspeare.gettops()
for l in lstwords:
    print(l)

print("\nSmall Shakespeare")
s, l = parseFile('data/smallspeare.txt')
fromt, tot, ll = todict(s, l)
smallspeare = HMM(len(fromt), fromt, tot, k=5)
smallspeare.learn(ll, tol=0.1, prt=True)
lstwords = smallspeare.gettops()
for l in lstwords:
    print(l)
