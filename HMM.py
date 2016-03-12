#!/usr/bin/env python
# Implementation of HMM for sonnet generation
import numpy as np
ZERO = 1e-323                   # add to everything to make sure no log(0)'s
STARTSTATE = ''                 # start state
EOL = '\n'
def parseFile(FN, delim=" "):
    """
    Takes a filename and parses into (set, list) tuple. Simplest tokenization
    Input:
        string - filename to be parsed
    Output:
        set - set of tokens in file
        list - list of tokenized inputs; each line assumed to be one input
    """
    tokens = set()
    tokens.add(STARTSTATE)
    tokenlist = list()
    f = open(FN, 'r')
    for l in f.readlines():
        line = l.split(delim)   # split by delim
        for i in line:
            tokens.add(i.strip())
        tokens.add(EOL)         # manually add EOL since dropped by strip
        tokenlist.append([STARTSTATE] + 
                [i.strip() for i in line] + 
                [EOL])
    return tokens, tokenlist

def todict(tokens, tokenlist):
    """
    Turns set of tokens + list of lists of tokens into dict from tokens to
    indicies and list of indicies
    Input:
        set - set of tokens
        list - list of lists of tokens, corresponding to each input instance
    Output:
        dict - mapping from tokens to indicies
        dict - mapping from indicies to tokens
        list - list of list of indicies, corresponding to each input instance
    """

    totokendict = dict()
    fromtokendict = dict()
    retlist = list()
    # first build dicts
    for index, token in zip(range(0, len(tokens)), tokens):
        totokendict[index] = token
        fromtokendict[token] = index
    # build return list
    for l in tokenlist:
        line = list()
        for c in l:
            line.append(fromtokendict[c])
        retlist.append(line)
    return fromtokendict, totokendict, retlist

class HMM(object):
    """
    Represents an HMM
    Can do unsupervised learning and Viterbi

    Member variables:
        int N       - Number tokens
        int k       - Number hidden states (default 5)
        float lamb  - Regularization (default 0)
        nparray A   - k * k transition matrix (default 1/k everything)
                    - FROM second index TO first index (columns sum to 1)
        nparray O   - k * N emission matrix (default 1/k everything)
        fromtoken   - dict from tokens to indicies
        totoken     - dict from indicies to tokens
    """
    def __init__(self, N, fromtoken, totoken, k=5, lamb=0, A=None, O=None):
        super(HMM, self).__init__()
        self.N = N
        self.k = k + 2                  # start, end states
        self.lamb = lamb
        self.fromtoken = fromtoken
        self.totoken = totoken
        self.setA(A)
        self.setO(O)
    def setA(self, A = None):
        """
        sets A matrix, k * k (if passed None, sets to 1/k)
        """
        if A is None:
            self.A = np.zeros([self.k, self.k]) + 1.0 / self.k
        else:
            self.A = np.array(A) + max(0, ZERO - self.A.min())
    def setO(self, O = None):
        """
        sets O matrix, k * N (if passed None, sets to 1/k)
        """
        if O is None:
            self.O = np.zeros([self.k, self.N]) + 1.0 / self.k
        else:
            self.O = np.array(O) + max(0, ZERO - self.O.min())
    def predict(self, startindex=0, endindex=-1, log=False, max_iters=10,
            multiplier=None, renorm=True):
        """
        Runs Viterbi and predicts max sequence
        A_ij = prob transition from i to j
        Inputs:
            int startindex  - index of start state, default 0
            int endindex    - index of end state, default -1
            func agg        - function to aggregate, default np.add (log
                likelihoods)
            bool log        - use log probabilities
            int max_iters   - max number of tokens to predict before truncating
            list(float)     - external probability multiplier on tokens
                multiplier      (possibly from other HMMs). Default: all 1s
            bool renorm     - whether to renormalize likelihoods at each step
                                only relevant when not using log probs
        Outputs:
            float           - probability
            list(int)       - maximum probability sequence, in state number
        """
        if self.A is None or self.O is None:
            raise ValueError("A or O is None")
        if multiplier is None:
            multiplier = np.array([1] * self.k)
        else:
            multiplier = np.array(multiplier) + max(0, ZERO - min(multiplier))

        log_likelihoods = list()        # list of k floats, likelihoods at each
                                        # step
        paths = list([list([startindex])] * self.k)  
            # list of k paths, paths for each likelihood

        # set up starting likelihood
        log_likelihoods.append(np.zeros(self.k) + ZERO)
        log_likelihoods[0][startindex] = 1  # start state has 1 probability at
                                            # first
        if log == True:
            log_likelihoods[0] = np.log(log_likelihoods[0])
        for it in range(max_iters):
            likelihoods = np.zeros(self.k)
            newpaths = list([0] * self.k)  
            for j in range(self.k):
                # prob transition from i into j
                if log == True:
                    probabilities = np.add(np.log(self.A)[j], 
                            log_likelihoods[-1]) + np.log(multiplier[j])
                else:
                    probabilities = np.multiply(self.A[j], 
                            log_likelihoods[-1]) * multiplier[j]
                index = probabilities.argmax()  # argmax_i P(i -> j)
                # store likelihood, path for maximum
                likelihoods[j] = probabilities[index]
                newpaths[j] = paths[index] + [j]
            if log == True:
                log_likelihoods.append(likelihoods)
            else:
                log_likelihoods.append(likelihoods / sum(likelihoods))
            paths = newpaths
        index = log_likelihoods[-1].argmax()
        return log_likelihoods[-1][index], paths[index]
    def calcA(self, seq):
        """
        computes alpha values
        Each column of alpha is normalized since it only makes sense this way
        Input:
            seq - M-length input sequence, already fromtoken'd
        Output:
            a   - M x k alpha values
        """
        M = len(seq)
        k = self.k
        a = np.zeros([M, k])
        a[0, seq[0]] = 1
        for z in range(1, M):
            a[z] = np.multiply(self.O[:, seq[z]],
                    np.dot(self.A, a[z-1]))
            a[z] /= a[z].sum()
        return a
    def calcB(self, seq):
        """
        computes beta values
        Each column of beta is normalized since it only makes sense this way
        Input:
            seq - M-length input sequence, already fromtoken'd
        Output:
            b   - M x k beta values
        """
        M = len(seq)
        k = self.k
        b = np.zeros([M, k])
        b[-1, -1] = 1          # start B at end state
        for z in range(1, M):
            b[-z - 1] = np.dot(np.transpose(self.A), 
                    np.multiply(self.O[:, seq[-z]],b[-z]))
            b[-z - 1] /= b[-z - 1].sum()
        return b
    def EM(self, seq):
        """
        performs expectation-maximization
        Input:
            seq - M-length input sequence, already fromtoken'd
        Output:
            resA, resO - Frobenius norms of newA - A, newO - O
        """
        A = np.zeros(self.A.shape)
        O = np.zeros(self.O.shape)
        alpha = self.calcA(seq)
        beta = self.calcB(seq)

        # O update rule
        for state in range(self.k):
            for token in range(self.N):
                num = 0
                den = 0
                for i, z in enumerate(seq):
                    marginal = alpha[i, state] * beta[i, state] /\
                            np.dot(alpha[i], beta[i])
                    if z == token:
                        num += marginal
                    den += marginal
                O[state, token] = num / den
        # A update rule
        for state1 in range(self.k):
            for state2 in range(self.k):
                num = 0
                den = 0
                for i, z in enumerate(seq[1:]):
                    # num is properly normalized, sums to 1 over states
                    num += alpha[i, state1] * beta[i + 1, state2] * \
                            self.A[state2, state1] * self.O[state2, z] / \
                            np.dot(np.dot(self.A, alpha[i]), np.multiply(
                                beta[i + 1], np.transpose(self.O)[z]))
                    den += alpha[i, state1] * beta[i, state1] /\
                            np.dot(alpha[i], beta[i])
                A[state2, state1] = num / den
        resA = np.sqrt(((self.A - A)**2).sum())
        resO = np.sqrt(((self.O - O)**2).sum())
        self.A = A
        self.O = O
        return resA, resO
