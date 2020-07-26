# -*- coding: utf-8 -*-
"""
Created on Sun May  3 00:13:23 2020

@author: ISIL
"""

import requests
import bs4

discard_elements = ['html','www','[',']','=','!','+','$','&','/','{','(',')','}','*',
                    '|','-','@','€','.',',',';','<','>'," ",'\n']

file_training = open("articles.txt","w") 
file_test = open("test_articles.txt","w") 

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
    

def writeToFile(url,file):

    res = requests.get(url)
    
    wiki = bs4.BeautifulSoup(res.text,"lxml")
    elems = wiki.select('p')
    
    sentences = []
    
    for i in range(len(elems)):
        text = elems[i].getText()
        sentence = text.split(".") #seperating text into sentences
        for j in range (len(sentence)):
            sentences.append(sentence[j])
    
    
    #loop for whole text
    for i in range(len(sentences)):  
        
        sentence = sentences[i]
        word_list = sentence.split(" ")
        
        #loop for one sentence
        for j in range (len(word_list)):
            word = word_list[j]
            isDiscard = False
            
            for k in range(len(discard_elements)):
                if(discard_elements[k] in word):
                    isDiscard = True
                
            if (not isDiscard and not RepresentsInt(word)):
                word = (str(word.encode('utf8')))[2:-1]
                file.write(word)
                file.write(",")
        file.write("\n")
        

f = open("urls.txt", "r")
lines = f.readlines()

for i in range(len(lines)):
    url = (lines[i])[:-1]
    writeToFile(url,file_training)
    file_training.write("\n______\n")

f.close()
file_training.close()
  
f_ = open("test_urls.txt", "r")
lines_ = f_.readlines()

for i in range(len(lines_)):
    url = (lines_[i])[:-1]
    writeToFile(url,file_test)
    file_test.write("\n______\n")
    
f_.close()
file_test.close()