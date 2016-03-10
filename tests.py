#!/usr/bin/env python
from HMM import *
def test(a, b, c):
    """
    if a is true, print "SUCCESS:" + b 
        else print "ERROR:" + c
    """
    if a:
        print("SUCCESS: " + b)
    else:
        print("ERROR: " + c)

if __name__ == '__main__':
    # parseFile()
    tokens, tokenlist = parseFile('data/dummy.txt')
    test(tokens == set(['0', '1', '\n', '']),
            "parseFile has correct tokens",
            "parseFile has incorrect tokens")
    # todict()
    fromtoken, totoken, dat = todict(tokens, tokenlist)
    test(set(fromtoken.keys()) == set(['0', '1', '\n', '']),
            "todict() fromtoken has correct keys",
            "todict() fromtoken has incorrect keys")
    test(all([totoken[fromtoken[i]] == i for i in fromtoken.keys()]) and
            all([fromtoken[totoken[i]] == i for i in totoken.keys()]),
            "todict() dicts match",
            "todict() dicts don\'t match")
    # viterbi
    k = 2
    testHMM = HMM(4, fromtoken, totoken, k=k)
    testHMM.setA(np.array(
        [[0, 0.51, 0.49, 0],
            [0, 0.1, 0.9, 0],
            [0, 0.9, 0.1, 0],
            [0, 0, 0, 1]]) + ZERO) # four states are ['', '0', '1', '\n']
    O = np.zeros([k + 2, len(totoken.keys())]) + ZERO
    O[fromtoken[''], 0] = 1
    O[fromtoken['0'], 1] = 1
    O[fromtoken['1'], 2] = 1
    O[fromtoken['\n'], 3] = 1       # each state just emits itself
    testHMM.setO(O)
    prob, seq = testHMM.predict()
    problog, seqlog = testHMM.predict(log=True)
    test(seqlog == seq,
            "Viterbi log matches multiply",
            "Viterbi log fails to match multiply")
