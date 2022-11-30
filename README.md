# AIIAP_personal_project

  1、將train.p 、  valid.p、 test.p、 放在同一個資料夾下
  2、讀入資料並分為train、valid、test：
  
    import pickle
    
    training_file = 'train.p'
    validation_file='valid.p'
    testing_file = 'test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']
    
    #這裡的X_train、X_valid、X_test為 (len(), 32, 32, 3) 的 numpy array
    
  
  3、將資料打散：
    from sklearn.utils import shuffle
    X_train, y_train = shuffle(X_train, y_train)
  
  4、將資料從區間 0-255 normalize成 0-1
    X_train = (X_train / 255.).astype('float32')
    X_valid = (X_valid / 255.).astype('float32')
    X_test  = (X_test  / 255.).astype('float32')
    
    #資料大小應為
    #(34799, 32, 32, 3)
    #(4410, 32, 32, 3)
    #(12630, 32, 32, 3)
    
    
  5、以keras API建立sequential model：
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import layers

    model = Sequential()
    model.add(layers.Conv2D(filters = 6, kernel_size = (5, 5), strides = (1, 1), padding ='valid', activation = 'relu', data_format = 'channels_last', input_shape = (32, 32, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation = 'relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(120, activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(84, activation = 'relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(43, activation = 'softmax'))
    print(model.summary()) #會印出model
    
    
    #相較原本的LeNet5 多新增兩層drop out layer，避免over fitting
    
    
  6、compile and fit：
    optm = Adam(lr = 0.002) 
    # 設定optimizer為Adam， learning rate = 0.002(可以自己調整)
    
    model.compile(optimizer = optm, loss= 'sparse_categorical_crossentropy', metrics = ['acc'] )
    # compile各項參數， 都可以上網查查再來調整
    
    train_history = model.fit(X_train, y_train, batch_size = 512, epochs= 30, validation_data = (X_valid, y_valid))
    # 開train並記錄各項結果
    
    history_dict = train_history.history
    history_dict.keys()
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs_ = range(1,len(acc)+1)

    plt.plot(epochs_ , loss , label = 'training loss')
    plt.plot(epochs_ , val_loss , label = 'val los')
    plt.title('training and val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(epochs_ , acc , label='train accuracy')
    plt.plot(epochs_ , val_acc , label = 'val accuracy')
    plt.title('train and val acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.show()
    #畫出train accuracy、validation accuracy 以及 train loss、validation loss
    
    score = model.evaluate(X_test, y_test, verbose=0.1)
    print('Test loss', score[0])
    print('Test accuracy', score[1])
    #印出test loss、 test accuracy
  
