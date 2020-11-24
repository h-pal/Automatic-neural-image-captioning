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
