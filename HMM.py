#!/usr/bin/env python
# Implementation of HMM for sonnet generation
import numpy as np
ZERO = 1e-323                   # add to everything to make sure no log(0)'s
STARTSTATE = ''                 # start state
EOL = '\n'
import bisect                   # for random prediction
import math                     # math.isnan()
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
        l = l.strip(".,?!'\"")
        line = l.split(delim)   # split by delim
        for i in line:
            tokens.add(i.strip().lower())
        tokens.add(EOL)         # manually add EOL since dropped by strip
        tokenlist.append([STARTSTATE] + 
                [i.strip().lower() for i in line] + 
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
        forbid transitions out of end state, into start state
        """
        if A is None:
            self.A = np.zeros([self.k, self.k]) + 1.0 / (self.k - 1)
            endtrans = np.zeros(self.k) + ZERO
            endtrans[-1] = 1
            self.A[0] = 0
            self.A[:, -1] = endtrans
            # randomly vary A small-ly while preserving sum(columns) = 1
            # break degeneracies (observed some in testing)
            for i in range(1, self.k - 1):
                randarr = np.random.rand(self.k - 1) / (5 * self.k)
                randarr -= randarr.mean()
                self.A[1: , i] += randarr
        else:
            self.A = np.array(A) + max(0, ZERO - A.min())
    def setO(self, O = None):
        """
        sets O matrix, k * N (if passed None, sets to 1/k)
        start state is k=0, end state is k=-1, must have fixed emission
        probabilities
        """
        if O is None:
            self.O = np.zeros([self.k, self.N]) + 1.0 / self.k
            startemis = np.array([1] + [ZERO] * (self.N - 1))
            endemis = np.zeros(self.N) + ZERO
            endemis[self.fromtoken[EOL]] = 1
            self.O[0, :] = startemis
            self.O[-1, :] = endemis
            # randomly vary A small-ly while preserving sum(rows) = 1
            # break degeneracies (observed some in testing)
            for i in range(1, self.k - 1):
                randarr = np.random.rand(self.N) / (5 * self.k)
                randarr -= randarr.mean()
                self.O[i , :] += randarr
        else:
            self.O = np.array(O) + max(0, ZERO - O.min())
    def predict(self, max_iters=-1, mult_state=None, rand=False,
            retl=False):
        """
        Runs Viterbi and predicts max sequence
        A_ij = prob transition from i to j
        Inputs:
            int max_iters   - max number of tokens to predict before truncating
                                Default: -1 (no truncation)
            list(float)     - external probability multiplier on tokens
                mult_state      (possibly from other HMMs). Default: all 1s
            bool rand       - make random choices per probability distribution
                                instead of random weights
            bool retl       - whether to return log-likelihood. Default False
        Outputs:
            float           - probability
            list(int)       - maximum probability sequence, in state number
        """
        startindex = 0
        endindex = self.k - 1

        if self.A is None or self.O is None:
            raise ValueError("A or O is None")
        if mult_state is None:
            mult_state = np.array([1] * self.k)
        else:
            mult_state = np.array(mult_state) + max(0, ZERO - min(mult_state))

        mult_emis = np.array([1] * self.k)

        tot_likelihoods = list()        # list of k floats, likelihoods at each
                                        # step
        paths = list([list([STARTSTATE])] * self.k)  
            # list of k paths, paths for each likelihood

        # set up starting likelihood
        tot_likelihoods.append(np.zeros(self.k) + ZERO)
        tot_likelihoods[0][startindex] = 1  # start state has 1 probability at
                                            # first
        tot_likelihoods[0] = np.log(tot_likelihoods[0])

        it = 0 # most natural way for -1 = infinite loop is this
        maxpath = paths[0]
        while it != max_iters and maxpath[-1] != EOL:
            index = self._predhelper(tot_likelihoods, paths, mult_state,
                    mult_emis, rand)
            maxpath = paths[index]
            it += 1
        if retl:
            return tot_likelihoods[-1][index], maxpath if EOL not in maxpath\
                    else maxpath[ : maxpath.index(EOL) + 1]
        else:
            return maxpath if EOL not in maxpath else \
                    maxpath[ : maxpath.index(EOL) + 1]
    def _predhelper(self, tot_likelihoods, paths, mult_state, mult_emis, rand):
        likelihoods = np.zeros(self.k)
        newpaths = list([0] * self.k)
        for j in range(self.k):
            # prob transition from i into j
            probabilities = np.add(np.log(self.A)[j],
                    tot_likelihoods[-1]) + np.log(mult_state[j])
            if rand == True:    # use partial sum trick to choose a case
                                # when rand == True
                sumProbs = list([0])
                newprobs = probabilities - max(probabilities)
                for p in newprobs:
                    sumProbs.append(sumProbs[-1] + np.exp(p))
                if all(np.array(sumProbs) == 0):
                    index = 0
                else:
                    index = bisect.bisect(sumProbs,
                            np.random.rand() * sumProbs[-1]) - 1
            else:
                index = probabilities.argmax()  # argmax_i P(i -> j)
            # store likelihood, path for maximum
            likelihoods[j] = probabilities[index]
            if rand == True:
                ind_token = bisect.bisect(np.cumsum(self.O[j, :]),
                        np.random.rand() * sum(self.O[j, :]))
            else:
                ind_token = self.O[j, :].argmax()
            newpaths[j] = paths[index] + [self.totoken[ind_token]]
        tot_likelihoods.append(likelihoods)
        for i in range(len(paths)):
            paths[i] = newpaths[i]
        if rand == True:
            newprobs = tot_likelihoods[-1] - max(tot_likelihoods[-1])
            sumProbs = list([0])
            for p in newprobs:
                sumProbs.append(sumProbs[-1] + np.exp(p))
            r = np.random.rand() * sumProbs[-1]
            index = bisect.bisect(sumProbs, r) - 1
        else:
            index = tot_likelihoods[-1].argmax()
        return index
    def toktostr(self, s, delim=" "):
        """
        turns a sequence of tokens into tokenstring
        """
        # do not convert start state
        return delim.join(s[1: -1]) + EOL
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
                    np.multiply(self.O[:, seq[-z]] ,b[-z]))
            b[-z - 1] /= b[-z - 1].sum()
        return b
    def EM(self, seqs):
        """
        performs expectation-maximization
        Input:
            seqs - list of input sequences, already fromtoken'd
        Output:
            resA, resO - Frobenius norms of newA - A, newO - O
        """
        A = np.array(self.A)
        O = np.array(self.O)
        alphas = list()             # cache computation of a, b to save time
        betas = list()

        # O update rule
        for state in range(1, self.k - 1):
            for token in range(self.N):
                num = 0
                den = 0
                for s, seq in enumerate(seqs):
                    if len(alphas) != len(seqs): # not yet precomputed
                        alpha = self.calcA(seq)
                        beta = self.calcB(seq)
                        alphas.append(alpha)
                        betas.append(beta)
                    else:
                        alpha = alphas[s]
                        beta = betas[s]
                    for i, z in enumerate(seq):
                        if np.dot(alpha[i], beta[i]) == 0:
                            print(self.A.round(3))
                            print()
                            print(self.O.round(3))
                            print()
                            print(alpha.round(3))
                            print()
                            print(beta.round(3))
                            print()
                            exit(0)
                        marginal = alpha[i, state] * beta[i, state] /\
                                np.dot(alpha[i], beta[i])
                        if z == token:
                            num += marginal
                        den += marginal
                O[state, token] = num / den
        # A update rule
        for state1 in range(self.k - 1):
            for state2 in range(1, self.k):
                num = 0
                den = 0
                for s, seq in enumerate(seqs):
                    alpha = alphas[s]
                    beta = betas[s]
                    for i, z in enumerate(seq[1:]):
                        # num is properly normalized, sums to 1 over states
                        if np.dot(np.dot(self.A, alpha[i]), np.multiply(
                                    beta[i + 1], np.transpose(self.O)[z])) == 0:
                            continue # edge case? if one token not being emitted
                        else:
                            num += alpha[i, state1] * beta[i + 1, state2] * \
                                    self.A[state2, state1] * self.O[state2, z] / \
                                    np.dot(np.dot(self.A, alpha[i]), np.multiply(
                                        beta[i + 1], np.transpose(self.O)[z]))
                            den += alpha[i, state1] * beta[i, state1] /\
                                    np.dot(alpha[i], beta[i])
                A[state2, state1] = 0 if den == 0 else num / den
        resA = np.sqrt(((self.A - A)**2).sum())
        resO = np.sqrt(((self.O - O)**2).sum())
        self.setA(A)
        self.setO(O)
        return resA, resO
    def learn(self, seqs, tol=0.003, max_iter=200, prt=False):
        """
        runs EM until Frobenius norm is within tol / self.k. Default tol =
        0.01.
        Inputs:
            seq - same input to EM
            tol - threshold
            max_iter - maximum number of iterations to run EM, in case not
                    converging. default: 50
            prt - whether to print each iteration for debugging
        """
        resA, resB = self.EM(seqs)
        it = 0
        while max(resA, resB) > tol / self.k and it < max_iter:
            resA, resB = self.EM(seqs)
            if prt:
                print(it, (round(resA, 3), round(resB, 3)))
            it += 1
        return resA, resB
    def gettops(self, numWords=5, thresh=1e-10, unique=True):
        """
        returns words that correspond to top numWords weights for each latent
        state.
        Input:
            numWords - number of words to list for each latent state. Default: 5
            thresh - emission probability beyond which not considered top.
                    Default: 1e-10
            unique - so each token can only be assigned to one latent state
        Output:
            list(list(str)) - list of list of top numWords words for each token
        """
        O = np.array(self.O)
        if unique:
            for i in range(self.N):
                arr = np.zeros(self.k)
                arr[O[:, i].argmax()] = O[:, i].max()
                O[:, i] = arr
        l = list()
        for i in range(self.k):
            ranks = list(zip(O[i,:], range(self.N)))
            ranks.sort()
            ranks.reverse()
            lst = list()
            for i in ranks[:numWords]:
                if i[0] > thresh: # only add if it has a reasonable emission
                                  # probability
                    lst.append(self.totoken[i[1]])
            l.append(lst)
        return l
