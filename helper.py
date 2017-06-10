import codecs
def create_data(file_path,output_path):
    reader = codecs.open(file_path,'r',)
    data = []
    for line in reader.readlines():
        data.append(eval(line))
    reader.close()
    with codecs.open(output_path,'w','utf-8') as outfile:
        for sample in data:
            for one in sample:
                outfile.write(one[0]+'\t'+one[2]+'\n')
            outfile.write('\n')

def get_maxlen(file_path):
    lendic = {}
    reader = codecs.open(file_path, 'r', )
    for line in reader.readlines():
        l = len(eval(line))
        if l not in lendic:
            lendic[l] = 0
        lendic[l] += 1
    reader.close()
    print lendic

if __name__ == '__main__':
    # train_path = 'restaurant2016/train_data.txt'
    # output_path1 = 'restaurant2016/traindata'
    test_path = 'restaurant2016/test_data.txt'
    output_path2 = 'restaurant2016/testdata'
    # create_data(train_path, output_path1)
    create_data(test_path,output_path2)
    # get_maxlen(train_path)