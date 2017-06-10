from sklearn.preprocessing import LabelBinarizer
from itertools import chain
from sklearn.metrics import classification_report
def evalution(gold_path,pred_path):
    labels = ['TERM']
    gold_num_dic = {label: 0 for label in labels}
    pred_num_dic = {label: 0 for label in labels}
    corr_num_dic = {label: 0 for label in labels}
    precision = -1.0
    recall = -1.0
    f_score = -1.0
    gold_num = 0
    pred_num = 0
    corr_num = 0
    ygold = load_data(path=gold_path)
    ypred = load_data(path=pred_path)
    for i in range(len(ygold)):
        for j in range(len(ygold[i])):
            if ygold[i][j][0] == 'B':
                gold_num_dic[ygold[i][j][2:]] += 1
            if ypred[i][j][0] == 'B':
                pred_num_dic[ypred[i][j][2:]] += 1
            if ygold[i][j][0] == 'B' and ygold[i][j] == ypred[i][j]:
                k = j+1
                flag = True
                while k < len(ygold[i]):
                    if (ygold[i][k][0] == 'O' or ygold[i][k][0] == 'B') and (ypred[i][k][0] == 'O' or ypred[i][k][0] == 'B'):
                        break
                    if ygold[i][k] != ypred[i][k]:
                        flag = False
                        break
                    k += 1
                if flag:
                    corr_num_dic[ygold[i][j][2:]] += 1
                    # print 'id%d,%d'%(i,j)

    print "type\tprecision\trecall\tf1-score\tsupport\n"
    for label in labels:
        # print corr_num_dic[label]
        # print gold_num_dic[label]
        # print pred_num_dic[label]
        precision = corr_num_dic[label]*1.0/pred_num_dic[label]
        recall = corr_num_dic[label]*1.0/gold_num_dic[label]
        f1_score = 2 * (precision * recall) / (precision + recall)
        print label + '     ' + str(precision) + '      '+str(recall) + '       ' + str(f1_score) + '        ' + str(gold_num_dic[label])





def bio_classification_report(y_gold,y_pred):
    #y_gold: [[],[],[]]
    #y_pred:
    lb = LabelBinarizer()
    y_gold_combined = lb.fit_transform(list(chain.from_iterable(y_gold)))
    y_pred_combined = lb.fit_transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset,key=lambda tag: tag.split('-',1)[::-1])
    class_indices = {cls:idx for idx,cls in enumerate(lb.classes_)}
    return classification_report(
        y_gold_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset
    )
def load_data(path):
    list = []
    temp = []
    with open(path,'r') as infile:
        for line in infile:
            if '' == line.strip():
                list.append(temp)
                temp = []
            else:
                line = line.strip()
                # print line
                # print line.split()
                temp.append(line.split('\t')[1])
    return list
if __name__ == '__main__':
    gold_path = 'restaurant2015/evaluation_data'
    pred_path = 'restaurant2015/result'
    # ygold = load_data(path=gold_path)
    # ypred = load_data(path=pred_path)
    # print bio_classification_report(ygold,ypred)

    evalution(gold_path,pred_path)