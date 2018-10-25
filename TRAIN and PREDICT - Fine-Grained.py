#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import print_function

from keras.datasets import mnist

from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import keras.backend as K
from itertools import product
from functools import partial

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras import metrics
from keras.layers import TimeDistributed

from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras import utils 


# In[2]:


# allow growth
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
ktf.set_session(get_session())
# test if using GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[3]:


# read data
file=pd.read_excel('./for_fine_grained.xlsx')
texts= np.array(file['comment_seg'].tolist())


# In[8]:


x=pd.DataFrame()


# In[12]:


print(-7%3)


# In[8]:


#num_words = 10000
maxlen = 180  #mean:185, min:20, 25%:82, 50%:124, 75%:204 max:3262
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
token_dict=dict(zip(tokenizer.word_index.values(),tokenizer.word_index.keys()))
token_dict[0]=''

texts_tokenized=tokenizer.texts_to_sequences(texts)

texts_padded = sequence.pad_sequences(texts_tokenized, maxlen=maxlen)
texts_padded_str =texts_padded.astype(str)
for index, value in enumerate(texts_padded_str):
    for p,q in enumerate(value):
        texts_padded_str[index][p]=token_dict[int(q)].strip()

        

# In[ ]:





# In[6]:


# lista=[]
# a=[]
# for i in range(X_padded_NL.shape[0]):
#     for j in range(X_padded_NL.shape[1]):
#         if X_padded_NL[i][j]=='':
#             X_padded_NL[i][j]='None'
#         else:
#             X_padded_NL[i][j]=X_padded_NL[i][j]+'/O'


# In[7]:


#pd.DataFrame(X_padded_NL).to_csv('./tobetagged.csv',header=None,index=None)


# In[8]:


x=np.append(texts_padded[0:230],texts_padded[999:2499],axis=0)  # training data


# In[9]:


#y=pd.read_excel('../tagged.xlsx',header=None,index=None) # read manually tagged data


# In[10]:


# y_tagged=x.copy()
# for i in range(y.shape[0]):
#     for j in range(y.shape[1]):
#         #print(y.iloc[i][j])
#         y_tagged[i][j]=str(y.iloc[i][j][y.iloc[i][j].rfind('/')+1:])


# In[11]:


#pd.DataFrame(y_tagged).to_csv('./tagged.csv',header=None,index=None)


# In[12]:


y_tagged=pd.read_csv('./tagged.csv',header=None)


# In[13]:


# map each category to a number for further one-hot
dict_label={'B-FA':0,'B-FO':1,'B-N':2,'B-P':3,'B-PE':4,'B-PR':5,'B-S':6,'I-FA':7,'I-FO':8,'I-N':9,'I-P':10,'I-PE':11,'I-PR':12,'I-S':13,'None':14,'O':15}
y_tagged_num=np.array(y_tagged).copy()
for index, value in enumerate(np.array(y_tagged)):
    #print(index,value)
    for p,q in enumerate(value):
        #print(q)
        y_tagged_num[index][p]=dict_label[q]

y = utils.np_utils.to_categorical(y_tagged_num,num_classes=16)


# In[14]:


import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
class myMetrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        precision, recall, f_score, support = precision_recall_fscore_support(val_targ, val_predict,average='weighted')
        #_val_f1 = f1_score(val_targ, val_predict)
        #_val_recall = recall_score(val_targ, val_predict)
        #_val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(f_score)
        self.val_recalls.append(recall)
        self.val_precisions.append(precision)
        print('  val_precision:',precision,'  val_recall:', recall,'   val_f1:',f_score) 
        return
 
mymetrics = myMetrics()


# In[15]:


max_features = 20000
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32))#Embedded layer that uses 128 length vectors to represent each word
#model.add(Conv1D(filters=32,kernel_size=3, padding='same', activation='relu'))  
#model.add(MaxPooling1D(pool_size=2)) #pool_size:Integer, size of the max pooling windows.
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,return_sequences=True)) #LSTM layer with 128 memory units (smart neurons) 
model.add(TimeDistributed(Dense(16,activation='softmax')))


# In[16]:


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))


# In[78]:


y_more=np.append(y_tagged_num,y_tagged_num[230:730],axis=0)
x_more=np.append(x,texts_padded[999:1499],axis=0) 


# In[81]:


y_more=np.append(y_more,y_tagged_num[230:730],axis=0)
x_more=np.append(x_more,texts_padded[999:1499],axis=0)  


# In[98]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_more, y_more, test_size=0.4, random_state=22)
weight_of_sample=y_train.copy()
np.place(weight_of_sample,weight_of_sample!=15|14,80)
np.place(weight_of_sample,weight_of_sample==14,1)
np.place(weight_of_sample,weight_of_sample==15,1)
y_train = utils.np_utils.to_categorical(y_train,num_classes=16)
y_test = utils.np_utils.to_categorical(y_test,num_classes=16)


# In[99]:


# weight_of_sample=y_tagged_num.copy()
# np.place(weight_of_sample,weight_of_sample!=15|14,20)
# np.place(weight_of_sample,weight_of_sample==14,1)
# np.place(weight_of_sample,weight_of_sample==15,1)

model.compile(loss='categorical_crossentropy', optimizer='Adadelta',metrics=['accuracy'],sample_weight_mode='temporal')

model.fit(X_train, y_train,batch_size=32,epochs=3  ,sample_weight=weight_of_sample ,validation_data=(X_test,y_test))


# In[101]:


model.save('my_model_fine_grained.h5')


# In[33]:


from keras.models import load_model
model=load_model('./my_model_fine_grained.h5')


# In[34]:


model.summary()


# In[103]:


y_predicted=model.predict_classes(np.append(texts_padded[230:999],texts_padded[2499:],axis=0)) 


# In[121]:


#back to category
dict_label_back={0:'B-FA',1:'B-FO',2:'B-N',3:'B-P',4:'B-PE',5:'B-PR',6:'B-S',7:'I-FA',8:'I-FO',9:'I-N',10:'I-P',11:'I-PE',12:'I-PR',13:'I-S',14:'None',15:'O'}
y_predicted_label=np.full((y_predicted.shape[0],y_predicted.shape[1]),'Noneeee')
for index, value in enumerate(np.array(y_predicted)):
    #print(index,value)
    for p,q in enumerate(value):
        #print(q)
        y_predicted_label[index][p]=dict_label_back[q]


# In[133]:


all_y_tagged=np.append(np.append(np.append(y_tagged[0:230],y_predicted_label[0:769],axis=0),y_tagged[230:],axis=0),y_predicted_label[769:],axis=0)


# In[139]:


texts_padded_str_label=texts_padded_str.copy()
for i in range(texts_padded_str.shape[0]):
    for j in range(texts_padded_str.shape[1]):
        #print(y.iloc[i][j])
        texts_padded_str_label[i][j]=str(texts_padded_str[i][j])+'/'+all_y_tagged[i][j]


# In[142]:


pd.DataFrame(texts_padded_str_label).to_excel('./all_tagged_fine_grained.xlsx')


# In[18]:


file=pd.read_csv('./tagged.csv',header=None)


# In[22]:


unique, counts = np.unique(file.values, return_counts=True)



