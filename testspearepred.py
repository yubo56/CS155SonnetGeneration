#!/usr/bin/env python
import pickle, HMM
# f = open('full.pkl', 'rb')
# o = pickle.load(f)
# mult = [1] * o.k
# mult[-1] = 0
# print(o.toktostr(o.predict(rand=True)))

f = open('full10.pkl', 'rb')
o = pickle.load(f)
mult = [1] * o.k
mult[-1] = 0
print(o.toktostr(o.predict(rand=True)))
