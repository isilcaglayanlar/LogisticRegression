# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:52:38 2020

@author: ISIL
"""
import pandas as pd
import POS_tag as pos
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np


#initialization
NUM_OF_FEATURES = 7
NUM_OF_ITERATIONS = 700
LEARNING_RATE = 0.0003 #alpha

articles = []
d = pd.read_csv('subjclueslen1-HLTEMNLP05.txt') #downloaded from http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/

weak_subjectives = []
positive_words = []

abstract_words = []
concrete_words = []

x = []
y = []

#preperation of files
def abstractAndConcreteWords():
    ab_f = open("100-400.txt", "r")
    ab_lines = ab_f.readlines()
    for i in range (len(ab_lines)):
        line = ab_lines[i]
        word_name = (line.split(" "))[0]
        abstract_words.append(word_name)
        
    con_f = open("400-700.txt", "r")
    con_lines = con_f.readlines()
    for i in range (len(con_lines)):
        line = con_lines[i]
        word_name = (line.split(" "))[0]
        concrete_words.append(word_name)

#weak words list 
def getSubjectiveLexicon():
   
    values = d.values    
   
    lexicon_list = []
    NF = 6
    
    for i in range(len(values)):
        
        features = []
        line = values[i][0]
        
        for i in range (NF-1):
            feature = line[(line.index("=")+1):(line.index(" "))]
            features.append(feature)
            line = line[(line.index(" "))+1:]
            
        feature = line[(line.index("=")+1):]
        features.append(feature)
        
        lexicon_list.append(features)
        
    return lexicon_list

def W_SSubjectivesList():
    subjective_lexicon = getSubjectiveLexicon()
    
    for i in range(len(subjective_lexicon)):
        word = subjective_lexicon[i][2]
        if(subjective_lexicon[i][0] == "weaksubj"):
            weak_subjectives.append(word)
        
        
    

def getWordData(word):
    
    wordInfo=[]
    if word.strip():

        word = str(word).lower() #since upper/lower case affects the intensity, upper is used
        
        wordInfo.append(word)
        
        token = nltk.word_tokenize(word) 
        POS = nltk.pos_tag(token)
        chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>}"""
        chunkParser = nltk.RegexpParser(chunkGram)
    
        chunked = chunkParser.parse(POS)
        POS_c = chunked[0][1]
        
        if (POS_c != "``"):
            wordInfo.append(POS_c)
        
    
        if(word in weak_subjectives):
           wordInfo.append(True)
        else:
            wordInfo.append(False)
            
            
        sia = SentimentIntensityAnalyzer()
        sps = sia.polarity_scores(word)
        compound = sps.get("compound")
            
        positive_polarity = False
        if(compound > 0):
            positive_polarity = True
                
        wordInfo.append(positive_polarity)
        word_u = word.upper()
        if(word_u in abstract_words):
            wordInfo.append("A") #abstract
        elif(word_u in concrete_words):   
            wordInfo.append("C") #concrete
        else :
            wordInfo.append("I") #invalid (neither in abstract list nor in concete list)
            
            

    return wordInfo

    
def readArticle(lines):
    
    article = []
    isEnd = False
    for i in range (len(lines)):
        sentence = []
        if(lines[i] != "______\n"):
            sentence = (lines[i]).split(",")
            
            if (sentence != "______\n"):
                for j in range(len(sentence)):
                    word = sentence[j]
                    sentence.append(word)
            
                article.append(sentence)
        else:
            isEnd = True
            break
    if (isEnd):
        articles.append(article)
        readArticle(lines[(i+1):])
        
    
       
def getArticleData(article):
    article_data = []
    
    for i in range(len(article)):
        sentence = article[i]
        sentence_data = []
        
        
        for j in range(len(sentence)):
            word = sentence[j]
           
            wordData = getWordData(word)
            
            if(len(wordData)==5 and wordData[4] != "I"):
                sentence_data.append(wordData)
     
        
        article_data.append(sentence_data)

    return article_data       

#converting features to numeric data
def addToFeatureMatrix(article_data):
    
    x_y = []
    x = []
    y = []
    name = []
    
    """
    FEATURE 1: POS
    FEATURE 2: WEAK
    FEATURE 3: POSITIVE
    FEATURE 4: POS_BEFORE
    FEATURE 5: POS_AFTER
    FEATURE 6: WEAK_BEFORE
    FEATURE 7: WEAK_AFTER
    """
    
    words = []
    
    
  
    for i in range(len(article_data)):
        
        sentence = article_data[i]
        
       
        
        for j in range(len(sentence)):
            
            features = []
            word = sentence[j]
            word_u = word[0].upper()
            features.append(word_u)
            if(word[4] == "A"):
                features.append(0)
            elif(word[4] == "C"):
                features.append(1)
            
            
            POS = pos.getNumeric(word[1])
            features.append(POS)
            if(word[2] == True):
                features.append(1)
            else:
                features.append(0)
                
            if(word[3] == True):
                features.append(1)
            else:
                features.append(0)
        
           
            if(len(features)==5):  
                words.append(features)
                
            
    
    
    #Gathering all features
    for j in range(1,(len(words)-1)):
       
        word_feature = []
        word = words[j]
        name.append(word[0])
        word_feature.append(1)
        word_feature.append(word[2])
        word_feature.append(word[3])
        word_feature.append(word[4])
        
        pre_word = words[j-1]
        next_word = words[j+1]
        
        word_feature.append(pre_word[2])
        word_feature.append(next_word[2])
        
        word_feature.append(pre_word[3])
        word_feature.append(next_word[3])
        x.append(word_feature)
        y.append(word[1])
        
    
    
    x_y.append(x) 
    x_y.append(y)
    
    
    return x_y

#LOGISTIC REGRESSION

# g(z) = 1/(1+e^(-z))
def sigmoid_function(z):
    exponential = np.exp(-z)
    sig = 1.0/(1.0+exponential)
    return sig

# hΘ(x) = 1/(1+e^(-Θ.xT))
# z = (-ΘT.x) -> for this case z = (-Θ.xT)
def probability(x,theta):
    # THETA = (NUM_OF_FEATURES+1) x 1
    # x = m x (NUM_OF_FEATURES+1)
    
    z = np.matmul(x,theta) # z = m x 1
    
    return sigmoid_function(z) # hΘ(x)
    

# Cost(hΘ(x),y) = -log(hΘ(x)) if y=1, -log(1-hΘ(x)) if y=0
# J(Θ) = (1/m)*sum(i from 1 to m) ((-y(i)*log(hΘ(x(i))))-(1-y)*log(1-hΘ(x(i))))
def cost_function(x,y,theta):
    m = len(y)
    h = probability(x,theta) # hΘ(x)
    
    J1 = ((-1)*y)*np.log(h) # y = 1
    J2 = (1-y)*np.log(1-h)  # y = 0
    
    J_sub = J1-J2 #J_sub = m x 1
    #sum(i from 1 to m)
    J_sum = J_sub.sum()
    return J_sum/m
    

# x = m x (NUM_OF_FEATURES+1)
# y = m x 1
# THETA = (NUM_OF_FEATURES+1) x 1
    
# gradient descent WITHOUT iteration
def gradient_descent(x,y,theta):
    m = len(y) # number of observations
    
    h = probability(x,theta) # hΘ(x)
    y_t = np.transpose(y)
    h_y = (h-y_t)
    x_t = np.transpose(x)
    g = np.matmul(x_t,h_y)
    g = LEARNING_RATE*(g/m)
    theta = theta - g
    return theta
    
def training(x,y,theta):
    for i in range(NUM_OF_ITERATIONS):
        theta = gradient_descent(x, y, theta)
    
        
    return theta

def decision_boundary(h):
    if(h>0.5):
        y=1
    elif(h<=0.5):
        y=0
    return y

def test(x,theta):
    
    y_predicted = []
    prob = probability(x,theta)
    for i in range(len(prob)):
        y = decision_boundary(prob[i])
        y_predicted.append(y)
    return y_predicted
        

def getX_Y(articles,theta):
    
    x_ = []
    y_ = []

    xy = []
        
    for i in range(len(articles)):
        article = articles[i]
        article_data = getArticleData(article)
        x_y = addToFeatureMatrix(article_data)

        
        x_ = x_ + x_y[0]
        y_ = y_ + x_y[1]
        
    #CONVERTING x and y arrays to matrices to make calculations easier
    m = len(x_)
        
    x = np.zeros([m,NUM_OF_FEATURES+1], dtype = int) 
    y = np.zeros([1,m], dtype = int) 
        
    for i in range(m):
        for j in range(NUM_OF_FEATURES+1):
            x[i][j] = x_[i][j]
    for k in range(m):
        y[0][k] = y_[k]
    
    xy.append(x)
    xy.append(y)

    return xy

def accuracy(y_test,y_predicted):
    
    total_counter = 0
    correct_counter = 0
    for i in range (len(y_test[0])):
        if(y_test[0][i] == y_predicted[i]):
            correct_counter += 1
        total_counter += 1
        
    return correct_counter/total_counter
            

hidden_layer_size = 4
learning_rate = 1
number_of_epochs = 1
C = 3 #window size
path = "./data" #use relative path like this


hidden_layer = np.zeros((1, hidden_layer_size))
print(hidden_layer[0][0])
#preperation of files
W_SSubjectivesList()
abstractAndConcreteWords()

#initialization of thetas as 1
THETA = np.ones([(NUM_OF_FEATURES+1), 1], dtype = float) 

#reading files that created by using article_file.py

#TRAINING        
f = open("articles.txt", "r")
lines = f.readlines()   
readArticle(lines)


xy_training = getX_Y(articles, THETA)
x_training = xy_training[0]
y_training = xy_training[1]

#minimized theta values
THETA = training(x_training,y_training,THETA)

articles.clear()

#TEST
f_test = open("test_articles.txt", "r")
lines_ = f_test.readlines()   
readArticle(lines_)

for i in range(len(articles)):
    xy_test = getX_Y(articles, THETA)
    x_test = xy_test[0]
    y_test = xy_test[1]
    
    y_predicted = test(x_test,THETA)
    
    accuracy = accuracy(y_test,y_predicted)