# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 00:46:32 2018

@author: Heller
"""

import os
import shutil
dire='train/'
to='test/'
c=0
for i in os.listdir(dire):
    c=0
    i=i+'/'
    for j in os.listdir(dire+i):
        c=c+1
        path2=os.path.join(to,i)
        shutil.move(dire+i+j,path2)
        if c==200:
            break
    