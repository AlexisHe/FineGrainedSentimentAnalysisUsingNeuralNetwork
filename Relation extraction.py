#!/usr/bin/env python
# coding: utf-8

# In[171]:


import pandas as pd
import re
import numpy as np
import time


# In[55]:


fine_grained=pd.read_excel('./all_tagged_fine_grained.xlsx',header=None)


# In[56]:


file=pd.DataFrame(fine_grained.apply(lambda x: ','.join(x), axis=1))


# In[68]:


file0=pd.concat([pd.Series(re.compile("[？。！\n]").split(row[0]))  for _, row in file.iterrows()]).reset_index() 


# In[69]:


file0.drop(labels=['index'],axis=1,inplace=True)


# In[74]:



file0=file0[file0[0].str.contains('(B-FA|B-FO|B-PE|B-PR|B-S|I-FA|I-FO|I-PE|I-PR|I-S|B-P|B-N|I-P|I-N)')]
file0.reset_index(inplace=True)
file0.drop(labels=['index'],axis=1,inplace=True)


# dict_label={'B-FA':0,'B-FO':1,'B-N':2,'B-P':3,'B-PE':4,'B-PR':5,'B-S':6,'I-FA':7,'I-FO':8,'I-N':9,'I-P':10,'I-PE':11,'I-PR':12,'I-S':13,'None':14,'O':15}
# 

# In[246]:


# entity tagging is accurate
# combine sentiment I and B tag

start = time.time()
pairs=pd.DataFrame(columns=['entity','entity_tag','sentiment','sentiment_tag'])
for i in range(file0.shape[0]):
    if(re.search('(B-FA|B-FO|B-PE|B-PR|B-S)',file0[0][i])):
        if(re.search('(B-P$|B-N|B-P,)',file0[0][i])):
            thestr=file0[0][i]
            thestrsplit=thestr.split(',')
            for j in range(len(thestrsplit)):
            #for j in thestr.split(','):
                if re.search('(B-FA|B-FO|B-PE|B-PR|B-S)',thestrsplit[j]):
#                     entity=thestrsplit[j].split('/')[0]
#                     if (j+1 < len(thestrsplit)):
#                         q=j+1
#                         while(re.search('(I-FA|I-FO|I-PE|I-PR|I-S)',thestrsplit[q])):
#                             entity+=thestrsplit[q].split('/')[0]
#                             q=q+1
#                             if q >= len(thestrsplit):
#                                 break
#                     pairs.loc[i,'entity']=entity
                    pairs.loc[i,'entity']=thestrsplit[j].split('/')[0]
                    pairs.loc[i,'entity_tag']=thestrsplit[j].split('/')[1]   
                if re.search('(B-P$|B-N)',thestrsplit[j]):
                    sentiment=thestrsplit[j].split('/')[0]
                    if (j+1 < len(thestrsplit)):
                        p=j+1
                        while(re.search('(I-P$|I-N)',thestrsplit[p])):
                            sentiment+=thestrsplit[p].split('/')[0]
                            #print(sentiment)
                            p=p+1
                            if p >= len(thestrsplit):
                                break
                    pairs.loc[i,'sentiment']=sentiment    
                    pairs.loc[i,'sentiment_tag']=thestrsplit[j].split('/')[1]  
                    
end=time.time()
print((end-start)/60)


# In[253]:


pairs.to_excel('./pairs.xlsx')

