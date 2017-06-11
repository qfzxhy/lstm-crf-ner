import numpy as np
import pandas as pd
import csv
import numpy as np
import re
def nextBatch(X,y,start_index,batch_size=128):
    last_index = start_index + batch_size
    X_batch = []
    y_batch = []
    if last_index > len(X):
        last_index = len(X)
    for i in range(start_index, last_index):
        X_batch.append(X[i])
        y_batch.append(y[i])
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch,y_batch

def nextRandomBatch(X,y,batch_size=128):
    X_batch = []
    y_batch = []
    index = np.arange(len(X))
    np.random.shuffle(index)
    X = X[index]
    for i in range(0,min(batch_size,len(X))):
        X_batch.append(X[i])
        y_batch.append(y[i])
    return np.array(X_batch),np.array(y_batch)

def buildMap(train_path):
    df_train = pd.read_csv(train_path,delimiter='\t',quoting=csv.QUOTE_NONE,skip_blank_lines=False,header=None,names=['word','label'])
    # print df_train
    # print df_train['word'][df_train['word'].notnull()]

    words = list(set(df_train['word'][df_train['word'].notnull()]))
    labels = list(set(df_train['label'][df_train['label'].notnull()]))

    word2id = dict(zip(words,range(1,len(words)+1)))
    label2id = dict(zip(labels,range(1,len(labels)+1)))

    id2word = dict(zip(range(1,len(words)+1),words))
    id2label = dict(zip(range(1, len(labels) + 1), labels))
    id2word[0] = "<PAD>"
    id2label[0] = "<PAD>"
    word2id["<PAD>"] = 0
    label2id["<PAD>"] = 0

    id2word[len(words)+1] = "<NEW>"
    id2label[len(labels)+1] = "<NEW>"
    word2id["<NEW>"] = len(words)+1
    label2id["<NEW>"] = len(labels)+1
    saveMap(id2word,id2label)
    return word2id,id2word,label2id,id2label

def saveMap(id2word,id2label):
    with open('data/word2id','w') as writer:
        for id in id2word.keys():
            writer.write(id2word[id] + "\t" + str(id) + "\r\n")
    with open('data/label2id','w') as writer:
        for id in id2label.keys():
            writer.write(id2label[id] + '\t' + str(id) + '\r\n')

def getTrainData(train_path,seq_max_len):
    word2id,id2word,label2id,id2label = buildMap(train_path)
    df_train = pd.read_csv(train_path,delimiter='\t',quoting=csv.QUOTE_NONE,skip_blank_lines=False,header=None,names=['word','label'])
    # print df_train
    df_train['word_id'] = df_train.word.map(lambda x : -1 if str(x) == str(np.nan) else word2id[x])
    df_train['label_id'] = df_train.label.map(lambda x : -1 if str(x) == str(np.nan) else label2id[x])
    # print list(df_train['word_id'])

    X,y = prepare(df_train['word_id'],df_train['label_id'],seq_max_len)
    # print X
    # print y
    num_samples = len(X)
    indexs = np.arange(num_samples)
    np.random.shuffle(indexs)
    X = X[indexs]

    y = y[indexs]
    return X,y
    #
    # if eval_path != None:
    #     X_train = X
    #     y_train = y
    #     X_eval,y_eval = getTestData(eval_path,seq_max_len,True)
    # else:
    #     X_train = X[:int(num_samples*0.8)]
    #     y_train = y[:int(num_samples*0.8)]
    #     X_eval = X[int(num_samples*0.8):]
    #     y_eval = y[int(num_samples*0.8):]
    #
    # return X_train,y_train,X_eval,y_eval
def getTrainEvalData(train_path,seq_max_len,eval_path=None):
    word2id,id2word,label2id,id2label = buildMap(train_path)
    df_train = pd.read_csv(train_path,delimiter='\t',quoting=csv.QUOTE_NONE,skip_blank_lines=False,header=None,names=['word','label'])
    # print df_train
    df_train['word_id'] = df_train.word.map(lambda x : -1 if str(x) == str(np.nan) else word2id[x])
    df_train['label_id'] = df_train.label.map(lambda x : -1 if str(x) == str(np.nan) else label2id[x])
    # print list(df_train['word_id'])

    X,y = prepare(df_train['word_id'],df_train['label_id'],seq_max_len)
    # print X
    # print y
    num_samples = len(X)
    indexs = np.arange(num_samples)
    np.random.shuffle(indexs)
    X = X[indexs]
    y = y[indexs]
    #
    if eval_path != None:
        X_train = X
        y_train = y
        X_eval,y_eval = getTestData(eval_path,seq_max_len,True)
    else:
        X_train = X[:int(num_samples*0.9)]
        y_train = y[:int(num_samples*0.9)]
        X_eval = X[int(num_samples*0.9):]
        y_eval = y[int(num_samples*0.9):]

    return X_train,y_train,X_eval,y_eval


def getTestData(test_path,seq_max_len,is_validation = True):
    word2id,id2word = loadMap('data/word2id')
    label2id,id2label = loadMap('data/label2id')
    #print word2id
    df_test = pd.read_csv(test_path,delimiter='\t',skip_blank_lines=False,header=None,quoting=csv.QUOTE_NONE,names=['word','label'])
    def mapfunc(x):
        if str(x) == str(np.nan):
            return -1
        elif x not in word2id:
            return word2id['<NEW>']
        else:
            return word2id[x]
    df_test['word_id'] = df_test.word.map(lambda x : mapfunc(x))
    df_test['label_id'] = df_test.label.map(lambda x : -1 if str(x) == str(np.nan) else label2id[x])
    if is_validation:
        X_test,y_test = prepare(df_test['word_id'],df_test['label_id'],seq_max_len)
        return X_test,y_test
    else:
        df_test['word'] = df_test.word.map(lambda x : -1 if str(x) == str(np.nan) else x)
        df_test['label'] = df_test.label.map(lambda x : -1 if str(x) == str(np.nan) else x)
        X_test,_ = prepare(df_test['word_id'],df_test['word_id'],seq_max_len)
        X_test_str,X_test_label_str = prepare(df_test['word'],df_test['label'],seq_max_len,is_padding=False)
        #print X_test_str
        return X_test,X_test_str,X_test_label_str
def loadMap(map_path):
    _2id = {}
    id2_ = {}
    with open(map_path,'r',) as reader:
        for line in reader.readlines():
            f = line.split('\t')[0]
            s = line.split('\t')[1]
            _2id[f] = int(s)
            id2_[int(s)] = f
    return _2id,id2_

def prepare(words_id,labels_id,seq_max_len,is_padding = True):
    X = []
    y = []
    x_temp = []
    y_temp = []
    for record in zip(words_id,labels_id):
        w = record[0]
        l = record[1]
        #line:-1
        if w == -1:
            if len(x_temp) < seq_max_len:
                X.append(x_temp)
                y.append(y_temp)
            x_temp = []
            y_temp = []
        else:
            x_temp.append(w)
            y_temp.append(l)

    if is_padding:
        X = padding(X,seq_max_len)
        y = padding(y,seq_max_len)
    return np.array(X),np.array(y)

def padding(samples,seq_max_len):
    n = len(samples)
    for i in range(n):
        if len(samples[i]) < seq_max_len:
            samples[i] = samples[i] + [0 for _ in range(seq_max_len-len(samples[i]))]
    return samples

def getEmbedding(emb_path):
    word2id,id2word = loadMap('data/word2id')
    with open(emb_path,'r') as reader:
        line_id = 1
        for line in reader:
            if line_id == 1:
                line = line.strip()
                dim = line.split(' ')[1]
                emb = np.zeros((len(word2id),int(dim)))
                has_lookup = len(word2id)*[0]
                line_id+=1
                continue
            else:
                infos = line.strip().split(' ')
                word = infos[0]
                if word in word2id:
                    emb[word2id[word]] = np.array(infos[1:],dtype=float)
                    has_lookup[word2id[word]] = 1
    for i in range(len(has_lookup)):
        if has_lookup[i] == 0:
            emb[i] = np.random.uniform(-0.25,0.25,int(dim))
    return emb

def reshape_to_batchnumsteps_numclasses(y,num_classes):
    new_ys = []
    for i in range(len(y)):
        new_y = []
        for j in range(len(y[i])):
            id = y[i][j]
            l = np.zeros((num_classes))

            l[id - 1] = 1
            new_y.append(l)
        new_ys.append(new_y)
    return np.array(new_ys)
def reshape_to_batch_numsteps(y,num_classes,num_steps):
    #y = [batch_size*num_steps,num_classes]
    #return [batch_size,num_steps]
    new_y = []
    temp = np.arange(1,num_classes+1)
    for y_aix1 in y:
        new_y.append(np.dot(y_aix1,temp))
    new_y = np.array(new_y)
    new_y = np.reshape(new_y,[-1,num_steps])
    return new_y
def extractEntity(sentence,labels):
    entitys = []
    pattern = re.compile(r'(B-[^BI]+)(I-[^BI]+)*')
    m = pattern.search(labels)
    while m:
        entity_label = m.group()
        # print entity_label
        label_start_index = labels.find(entity_label)
        label_end_index = label_start_index + len(entity_label)
        # print label_start_index
        # print label_end_index
        word_start_index = labels[:label_start_index].count('-') + labels[:label_start_index].count('O')
        word_end_index = word_start_index + entity_label.count('-')
        # print word_start_index
        # print word_end_index
        # print sentence[word_start_index:word_end_index]
        entitys.append(''.join(sentence[word_start_index:word_end_index]))
        labels = list(labels)
        labels[:label_end_index] = ['O' for _ in range(word_end_index)]
        labels = ''.join(labels)
        # print labels
        m = pattern.search(labels)
    return entitys
    # while entity:

if __name__ =='__main__':
    #getTrainData(train_path='data/traindata',eval_path='1',seq_max_len=20)
    #getTestData('data.word2id',20,True)
    # getTestData('data/testdata',20,False)
    #print getEmbedding('data/vectors')
    sentence = ['i','a','good','boy','ha']
    labels = 'OB-PESONB-TERMI-TERMI-TERM'
    # pattern = re.compile(r'(B-[^BI]+)(I-[^BI]+)*')
    # m = pattern.search(labels)
    # print m.group(0)
    print extractEntity(sentence,labels)
        # m = p.search('a1b2c3')

def save_test_data(X_test_str,X_test_label_str,output_path):
    with open(output_path,'w') as outfile:
        for i in range(len(X_test_str)):
            for j in range(len(X_test_str[i])):
                outfile.write(X_test_str[i][j] + "\t" + X_test_label_str[i][j] + '\n')
            outfile.write('\n')



