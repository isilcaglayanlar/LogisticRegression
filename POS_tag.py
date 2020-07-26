# -*- coding: utf-8 -*-
"""
Created on Mon May  4 01:31:54 2020

@author: ISIL
"""


def getNumeric(POS):
    POS_dict = {
        "CC" : 0.0,
        "CD" : 0.5,
        "DT" : 1.0,
        "EX" : 1.5,
        "FW" : 2.0,
        "IN" : 2.5,
        "JJ" : 3.0,
        "JJR" : 3.5,
        "JJS" : 4,
        "LS" : 4.5,
        "MD" : 5.0,
        "NN" : 5.5,
        "NNS" : 6.0,
        "NNP" : 6.5,
        "NNPS" : 7.0,
        "PDT" : 7.5,
        "POS" : 8.0,
        "PRP" : 8.5,
        "PRP$" : 9.0,
        "RB" : 9.5,
        "RBR" : 10.0,
        "RBS" : 10.5,
        "RP" : 11.0,
        "SYM" : 11.5,
        "TO" : 12.0,
        "UH" : 12.5,
        "VB" : 13.0,
        "VBD" : 13.5,
        "VBG" : 14.0,
        "VBN" : 14.5,
        "VBP" : 15.0,
        "VBZ" : 15.5,
        "WDT" : 16.0,
        "WP" : 16.5,
        "WP$" : 17.0,
        "WRB" : 17.5
    }


        
    numeric = POS_dict[POS]
    return numeric
      
            
            
            
            
            
            
            