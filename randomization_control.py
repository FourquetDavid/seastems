'''
Created on 10 dec. 2012

@author: davidfourquet
inspired by Telmo Menezes's work : telmomenezes.com
'''

'''
'''
import random

def new_constant():
    return abs(random.gauss(0,5))

def mutated_constant(constant):
    return abs(random.gauss(0,5)+constant)

  
