#!/usr/bin/env python
from HMM import *
def test(a, b, c, numTests):
    """
    if a is true, print "SUCCESS:" + b 
        else print "ERROR:" + c
    Return: [ [0,1] 1 ]
        sum together to get number tests succeeded/tried
    """
    if a:
        print("SUCCESS: " + b)
        numTests += np.array([1, 1])
    else:
        print("ERROR: " + c)
        numTests += np.array([0, 1])

if __name__ == '__main__':
    numTests = np.zeros(2)
    print('\nTESTING parseFile()')
    # parseFile()
    tokens, tokenlist = parseFile('data/dummy.txt')
    test(tokens == set(['0', '1', '\n', '']),
            "parseFile has correct tokens",
            "parseFile has incorrect tokens",
            numTests)
    print('\nTESTING todict()')
    # todict()
    fromtoken, totoken, dat = todict(tokens, tokenlist)
    test(set(fromtoken.keys()) == set(['0', '1', '\n', '']),
            "todict() fromtoken has correct keys",
            "todict() fromtoken has incorrect keys",
            numTests)
    test(all([totoken[fromtoken[i]] == i for i in fromtoken.keys()]) and
            all([fromtoken[totoken[i]] == i for i in totoken.keys()]),
            "todict() dicts match",
            "todict() dicts don\'t match",
            numTests)
    print('\nTESTING viterbi')
    # viterbi
    k = 2
    testHMM = HMM(4, fromtoken, totoken, k=k)
    testHMM.setA(np.array(
        [[0, 0.51, 0.49, 0],
            [0, 0.1, 0.9, 0],
            [0, 0.9, 0.1, 0],
            [0, 0, 0, 1]]) + ZERO) # four states are [STARTSTATE, '0', '1', EOL]
    O = np.zeros([k + 2, len(totoken.keys())]) + ZERO
    O[fromtoken[STARTSTATE], 0] = 1
    O[fromtoken['0'], 1] = 1
    O[fromtoken['1'], 2] = 1
    O[fromtoken[EOL], 3] = 1       # each state just emits itself
    testHMM.setO(O)
    prob, seq = testHMM.predict(max_iters=10)
    problog, seqlog = testHMM.predict(max_iters=10, log=True)
    test(seqlog == seq,
            "Viterbi log matches multiply",
            "Viterbi log fails to match multiply",
            numTests)
    test(seq == [0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "Viterbi sequence is correct",
            "Viterbi sequence is incorrect",
            numTests)
    probmult, seqmult = testHMM.predict(max_iters=10, multiplier=[0, 1, 0, 0],
            log=True)
    test(seqmult == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "With multiplier is correct",
            "With multiplier is incorrect",
            numTests)

    print('\n' + str(int(numTests[0])) + " tests out of " + 
            str(int(numTests[1])) + " tests passed!")
