#!/usr/bin/env python
from HMM import *
def test(a, b, c, numTests, ifFail=None):
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
        if ifFail != None:
            ifFail()

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
        [[0, 0, 0, 0],
            [0.51, 0.1, 0.9, 0],
            [0.49, 0.9, 0.1, 0],
            [0, 0, 0, 1]]) + ZERO) # four states are [STARTSTATE, '0', '1', EOL]
    O = np.zeros([k + 2, len(totoken.keys())]) + ZERO
    O[fromtoken[STARTSTATE], 0] = 1
    O[fromtoken['0'], 1] = 1
    O[fromtoken['1'], 2] = 1
    O[fromtoken[EOL], 3] = 1       # each state just emits itself
    testHMM.setO(O)
    prob, seq = testHMM.predict(max_iters=10)
    test(seq == [0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "Viterbi sequence is correct",
            "Viterbi sequence is incorrect",
            numTests)
    probmult, seqmult = testHMM.predict(max_iters=10, multiplier=[1, 1, 0, 1])
    test(seqmult == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "With multiplier is correct",
            "With multiplier is incorrect",
            numTests)
    
    print('\nTESTING alpha beta')
    # alphabeta
    k=2
    testHMM = HMM(4, fromtoken, totoken, k=k)
    a = testHMM.calcA(dat[0])
    b = testHMM.calcB(dat[0])
    # start state is defined at position [0], end at position [-1]
    test((a[0] == [1] + [0] * (len(b[-1]) - 1)).all() and 
            (a[1: ] == np.zeros([len(a) - 1, k + 2]) + 1 / (k + 2)).all(),
            "Right Alpha",
            "Wrong Alpha",
            numTests)
    test((b[-1] == [0.] * (len(b[-1]) - 1) + [1.]).all() and 
            (b[ :-1] == np.zeros([len(b) - 1, k + 2]) + 1 / (k + 2)).all(),
            "Right Beta",
            "Wrong Beta",
            numTests,
            lambda : print(b))

    print('\nTESTING EM')
    # EM
    k=2
    testHMM = HMM(4, fromtoken, totoken, k=k)
    testHMM.learn(dat, tol=0.001)
    test(testHMM.A.round(3).tolist() == [[0.0, 0.0, 0.0, 0.0], 
        [0.5, 0.389, 0.389, 0.0], 
        [0.5, 0.389, 0.389, 0.0], 
        [0.0, 0.222, 0.222, 1.0]],
        "A matrix correct",
        "A matrix incorrect",
        numTests,
        lambda : print(testHMM.A.round(3).tolist()))
    # O is not deterministic up to column shift, compare determinants
    test(abs(np.linalg.det(testHMM.O.round(3).tolist())) == 
            abs(np.linalg.det([[1.0, 0.0, 0.0, 0.0], 
            [0.0, 0.5, 0.0, 0.5],
            [0.0, 0.5, 0.0, 0.5], 
            [0.0, 0.389, 0.223, 0.389]])),
        "O matrix determinant correct",
        "O matrix determinant incorrect",
        numTests,
        lambda : print(testHMM.O.round(3).tolist()))

    print('\nRETESTING VITERBI')
    # Unsupervised prediction using Viterbi
    test(testHMM.predict(max_iters = 10, multiplier=[1,1,1,0])[1]
            == [0] + [1] * 10,
            "Deterministic Viterbi correct",
            "Deterministic Viterbi incorrect",
            numTests)
    print("Random Viterbi Test, should oscillate 1-2")
    for i in range(5):
        print(testHMM.predict(max_iters = 10, rand=True,
            multiplier=[1,1,1,0])[1])

    print('\n' + str(int(numTests[0])) + " tests out of " + 
            str(int(numTests[1])) + " tests passed!")
