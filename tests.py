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
    k=2
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

    # Unsupervised prediction using Viterbi
    print("Deterministic viterbi prediction")
    testHMM = HMM(4, fromtoken, totoken, k=k)
    testHMM.learn(dat, tol=0.001)
    predseq = testHMM.predict(max_iters=30)
    print(testHMM.toktostr(predseq))
    # random viterbi
    print("Random Viterbi Test, should oscillate 0-1 (half the time this works)")
    for i in range(5):
        teststr = testHMM.predict(rand=True, max_iters=30)
        print(testHMM.toktostr(teststr))

    print('\nTESTING DUMMY2')
    # Unsupervised prediction on new dummy file, should favor 1212 oscillations
    s, l = parseFile('data/dummy2.txt')
    fromt, tot, ll = todict(s, l)
    dummy1 = HMM(4, fromt, tot, k=2)
    dummy1.learn(ll)
    print("Short prediction, should oscillate")
    print(dummy1.toktostr(dummy1.predict(max_iters=30)))
    print("Long prediction, should prefer to terminate")
    print(dummy1.toktostr(dummy1.predict(max_iters=150)))
    print("Free prediction?")
    print(dummy1.toktostr(dummy1.predict()))

    print('\n' + str(int(numTests[0])) + " tests out of " + 
            str(int(numTests[1])) + " tests passed!")

    print('\nTESTING DUMMY3')
    # Third dummy file, testing whether can collapse 1234567895 to 1234512345
    s, l = parseFile('data/dummy3.txt')
    fromt, tot, ll = todict(s, l)
    dummy1 = HMM(len(fromt), fromt, tot, k=5)
    dummy1.learn(ll)
    print(dummy1.A.round(3))
    print(dummy1.O.round(3))
    print("Prediction deterministic")
    print(dummy1.toktostr(dummy1.predict(max_iters=30)))
    print("Prediction random")
    print(dummy1.toktostr(dummy1.predict(max_iters=30, rand=True)))
