from keras.layers import Dense, BatchNormalization,TimeDistributed, Embedding, LSTM, Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
import string
import re
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot
from num2words import num2words
from keras.applications import Xception

def expand_contractions(text):
  from contractions import  contractions_dict
  pattern = re.compile("({})".format("|".join( contractions_dict.keys())),flags = re.DOTALL| re.IGNORECASE)
    
  def replace_text(t):
    txt = t.group(0)
    if txt.lower() in  contractions_dict.keys():
      return  contractions_dict[txt.lower()]
        
  expand_text = pattern.sub(replace_text,text)
  return expand_text 

def num_to_words(text):
  text = text.split()
  for n, i in enumerate(text): 
    if(i.isnumeric()):
      text[n] = num2words(text[n])
  text = " ".join(text)
  return text

df = pd.read_csv(r'/content/captions.txt',delimiter=',')   
df['caption']=df['caption'].apply(lambda x: '<SOS> '+x+' <EOS>')

xcep = Xception()
xcep = Model(inputs = xcep.inputs,outputs = xcep.layers[-2].output)

for i in xcep.layers:
  i.Trainable = False
def prep_data(dir):
  for img in os.listdir(dir):
    if(img!='flickr30k_images' and img!='results.csv'):
      path = dir + '/' + img
      image = load_img(path,target_size=(299,299))
      image = img_to_array(image)
      image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
      image = image/255
      b=xcep.predict(image)
      yield img,b

sort = []
data=[]
for i,j in prep_data('/content/Images'):
  sort.append(i)
  data.append(j)

data = np.array(data)
data = data.reshape((data.shape[0],data.shape[2]))
data = np.repeat(data, repeats=5, axis=0)
data_train = data[:40330]
data_test = data[40330:]

sorterIndex = dict(zip(sort, range(len(sort))))
df.sort_values(by=['image'], key=lambda x: x.map(sorterIndex), ignore_index=True,inplace = True)

df['caption']=df['caption'].apply(lambda x: expand_contractions(x))
df['caption']=df['caption'].apply(lambda x: num_to_words(x))

tokenizer = Tokenizer(num_words = None,oov_token="<OOV>",filters=string.punctuation)
tokenizer.fit_on_texts(df['caption'][:40330])
word_index = tokenizer.word_index
sequences_train = tokenizer.texts_to_sequences(df['caption'][:40330])
sequences_train = np.array(sequences_train)
sequences_train = pad_sequences(sequences_train, maxlen=30, padding="pre", truncating="post")

x_train = np.array(sequences_train[:,:-1])
y_train = np.array(sequences_train[:,1:])
for i in y_train:
  for j in range(len(i)):
    if(i[j]==3):
      i[j] = 0
y_train = y_train.reshape((y_train.shape[0],y_train.shape[1],1))

size_of_vocabulary = len(word_index) + 1
embeddings_index = dict()
f = open(r'glove.6B.300d.txt',encoding="utf8")

for line in f:
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs

f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((size_of_vocabulary, 300))

for word, i in tokenizer.word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector

input = Input(batch_shape =(125,29) )
emb = Embedding(size_of_vocabulary,300,weights=[embedding_matrix],trainable=False)(input)
ls = LSTM(2048,stateful=True,return_sequences=True)(emb)
#bn1 = BatchNormalization()(ls)
#fc1 = TimeDistributed(Dense(512,activation='relu'))(bn1)
ou = TimeDistributed(Dense(1,activation='relu'))(ls)

model = Model(input,ou)
model.compile(loss='mean_squared_error',optimizer='RMSprop',metrics=[metrics.MeanAbsoluteError()])

 def gen_data():
  x = np.zeros((125, 29))
  y = np.zeros((125, 29,1))
  a=[]
  while True:
    for i in range(125):
      a.append(data_train[gen_data.current_idx].reshape((1,2048)))
      x[i] = x_train[gen_data.current_idx] 
      y[i] = y_train[gen_data.current_idx]
      gen_data.current_idx += 1
    K.set_value(model.layers[2].states[0],np.array(a).reshape((125,2048)))
    a=[]
    yield x, y
gen_data.current_idx = 0

for epoch in range(50):
    model.fit_generator(gen_data(), len(x_train)//125, 1,
                        validation_data=None, max_queue_size=1, shuffle=False)
    gen_data.current_idx = 0

def test():
  K.set_value(model.layers[2].states[0],data_test)
  test = np.full((125, 29), 3)
  b=[]

  for i in range(125):
    a = np.around(pred_ict(test)[i])
    for j in range(1,a.shape[0]):
      if(a[j-1]!=4):
        test[i][j] = a[j-1]
        a = np.around(pred_ict(test)[i])
    b.append(a)
  b = np.array(b)
  b = b.reshape((b.shape[0],b.shape[1]))

  reverse_word_index = dict(map(reversed, word_index.items()))
  reverse_word_index[0] = ''

  sentence =[]
  for i in range(b.shape[0]):
    text=''
    for j in range(b.shape[1]):
      text += reverse_word_index[b[i][j]]
      text += ' '
    sentence.append(text)

  real_images = []
  for i in df['image'][40330:]:
    path = '/content/Images/' + i
    image = load_img(path,target_size=(299,299))
    real_images.append(image)

  for i in range(0,125):
    display(real_images[i])
    print(sentence[i])
    
test()
