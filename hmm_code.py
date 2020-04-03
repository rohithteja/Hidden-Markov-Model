#pip install hmmlearn

"""Hidden Markov Models
Practical Session - M1 MLDM Rohith Teja MITTAKOLA
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from hmmlearn import hmm

#NOTE - The results are commented below the respective functions

"""Initializing the probability matrices"""

# DATA FOR THE MUSIC MODELING PROBLEM 

a = np.array([[1/3, 1/3, 1/3],
              [1/3, 1/3, 1/3],
              [1/3, 1/3, 1/3]]) #Transitional probability matrix

b = np.array([[1/2,  1/2, 0, 0, 0],
               [0, 1/2, 1/2, 0, 0],
               [0, 0, 0, 1/2, 1/2]]) #Observational probability matrix

pi = np.array([1/3, 1/3, 1/3]) #initial probabilities
A,B,C,D,E = 0,1,2,3,4 #observations
states = ['box1', 'box2', 'box3']
observations = ["A","B","C","D","E"]
sequence = np.array([A,B,E]) #given sequence of symbols
#sequence = np.array([A,B,C,D])

# TEST DATA FOR SOLVING THE BEACH MUSEUM PROBLEM FROM THE CLASS SLIDES
a = np.array([[0.2, 0.6, 0.2],
              [0.3, 0.2, 0.5],
              [0.1, 0.1, 0.8]])

b = np.array([[0.7,  0.3],
              [1, 0],
              [0.1, 0.9]])

pi = np.array([1, 0, 0])

states = ["Cloudy","Rainy","Sunny"]
beach = 0
museum = 1
observations = ["Beach","Museum"]
sequence = np.array([beach,beach,museum])

"""Forward Algorithm"""

def forward():
  global store_forward
  store_forward = np.zeros((sequence.shape[0],3))
  c1 = pi.T*b[:,sequence[0]]   #calculates the first column of the forward algorithm 
  store_forward[0, :] =c1
  for i in sequence[1:]:
    ctemp = [sum(c1*a[:,j]) for j in range(a.shape[0])]
    c2 = ctemp*b[:,i] #calculates the other columns recursively
    c1 = c2
    store_forward[np.where(sequence ==i), :] =c1
    cp = sum(c1)
  print("The calculation matrix is (columwise forward) \n", store_forward.T)
  print("The probability is ",sum(c1)) 
forward()

# The calculation matrix is (columwise forward) 
#  [[0.16666667 0.02777778 0.        ]
#  [0.         0.02777778 0.        ]
#  [0.         0.         0.00925926]]
# The probability is  0.009259259259259259

"""Backward Algorithm"""

def backward():
  global store_backward
  revsequence = sequence[::-1]
  clast = np.ones_like(a.shape[0]) #last column of backward algorithm
  store_backward = np.zeros((sequence.shape[0],3))
  for i in revsequence[0:len(revsequence)-1]: #recursion to find other columns
    ctemp = [sum(clast*a[j,:]*b[:,i]) for j in range(a.shape[0])]
    clast = ctemp
    cc = sum(pi*b[:,revsequence[-1]]*clast)
    store_backward[np.where(sequence ==i), :] = pi*b[:,revsequence[-1]]*clast #storing the values calculated
  store_backward[0:sequence.shape[0]-1,:] = store_backward[1:sequence.shape[0]+1,:]
  store_backward[-1,:] = np.ones_like(a.shape[0])
  print("The calculation matrix is (columwise backward) \n", store_backward.T)
  print("The probability is ",cc)
backward()

# The calculation matrix is (columwise backward) 
#  [[0.00925926 0.02777778 1.        ]
#  [0.         0.         1.        ]
#  [0.         0.         1.        ]]
# The probability is  0.009259259259259259

"""Viterbi Algorithm"""

def viterbi():
    store = np.zeros((sequence.shape[0],a.shape[0]))
    track = np.zeros((sequence.shape[0], a.shape[0]), 'int')
    store[0, :] = pi * b[:,sequence[0]] #calculating first column
    for t in range(1, sequence.shape[0]): #recursion
        for s2 in range(3):
            for s1 in range(3):
                score = store[t-1, s1] * a[s1, s2] * b[s2, sequence[t]] 
                if score > store[t, s2]: #checking max condition
                    store[t, s2] = score
                    track[t, s2] = s1
    follow = []
    follow.append(np.argmax(store[sequence.shape[0]-1,:]))
    for i in range(sequence.shape[0]-1, 0, -1):
        follow.append(track[i, follow[-1]])
    final=[states[i] for i in list(reversed(follow))]
    print("The calculation matrix is (columwise forward) \n", store.T)
    print("The probability is",max(store[sequence.shape[0]-1,:]))
    print("Optimal sequence:",final)

viterbi()

# The calculation matrix is (columwise forward) 
#  [[0.16666667 0.02777778 0.        ]
#  [0.         0.02777778 0.        ]
#  [0.         0.         0.00462963]]
# The probability is 0.004629629629629629
# Optimal sequence: ['box1', 'box1', 'box3']

"""Generate Sequence"""

def generate(number):
  choice =[]
  for i in range(number):
   box = np.random.choice(states, p= pi) #picks a random box from initial probability distribution
   symbol = np.random.choice(observations, p= b[states.index(box),:]) #picks symbol using
  #probability distribution of that symbol in respective box
   choice.append([box,symbol])
  print("The generated observations are \n ",choice)

generate(4) #generates 4 obs, can be changed to any number required

# The generated observations are 
#   [['box2', 'C'], ['box3', 'E'], ['box1', 'A'], ['box3', 'E']]

"""Log likehood forward algorithm"""

def logforward():
  store = np.zeros((sequence.shape[0],3))
  c1 = np.log(pi)+np.log(b[:,sequence[0]])  #calculates the first column of the forward algorithm 
  store[0, :] =c1
  for i in sequence[1:]:
    ctemp = [logsumexp(c1+np.log(a[:,j])) for j in range(a.shape[0])]
    c2 = ctemp+np.log(b[:,i]) #calculates the other columns recursively
    c1 = c2
    store[np.where(sequence ==i), :] =c1
    cp = logsumexp(c1)
  print("The calculation matrix is (columwise forward) \n", store.T)
  print("The negative likelihood is",cp)
  print("The probability is ",np.exp(cp))  

logforward()

# The calculation matrix is (columwise forward) 
#  [[-1.79175947 -3.58351894        -inf]
#  [       -inf -3.58351894        -inf]
#  [       -inf        -inf -4.68213123]]
# The negative likelihood is -4.68213122712422
# The probability is  0.009259259259259257

"""Log likehood backward algorithm"""

def logbackward():
  revsequence = sequence[::-1]
  clast = np.log(np.ones_like(a.shape[0])) #last column of backward algorithm
  store = np.zeros((sequence.shape[0],3))
  for i in revsequence[0:len(revsequence)-1]: #recursion to find other columns
    ctemp = [logsumexp(clast+np.log(a[j,:])+np.log(b[:,i])) for j in range(a.shape[0])]
    clast = ctemp
    cc = logsumexp(np.log(pi)+np.log(b[:,revsequence[-1]])+clast)
    store[np.where(sequence ==i), :] = np.log(pi)+np.log(b[:,revsequence[-1]])+clast #storing the values calculated
  store[0:sequence.shape[0]-1,:] = store[1:sequence.shape[0]+1,:]
  store[-1,:] = np.log(np.ones_like(a.shape[0]))
  print("The calculation matrix is (columwise backward) \n", store.T)
  print("The negative likelihood is",cc)
  print("The probability is ",np.exp(cc))

logbackward()

# The calculation matrix is (columwise backward) 
#  [[-4.68213123 -3.58351894  0.        ]
#  [       -inf        -inf  0.        ]
#  [       -inf        -inf  0.        ]]
# The negative likelihood is -4.68213122712422
# The probability is  0.009259259259259257

"""Checking results with "hmmlearn" python module"""

model = hmm.MultinomialHMM()
model.startprob_ = pi
model.transmat_ = a
model.emissionprob_ = b
model.fit
model.n_components = len(states)

model.predict(sequence.reshape(-1,1)) #same result as viterbi algorithm

# array([0, 0, 2])

model.score(sequence.reshape(-1,1)) #same as log probability calculated as above
#in logforward() and logbackward() functions
#this result raised to power "e" will give the probability obtained in forward and
#backward algorithms

# -4.68213122712422
#np.exp(-4.68213122712422)  = 0.009259259259259257
