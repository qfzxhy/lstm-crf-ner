import numpy as np
def reshape_to_batchnumsteps_numclasses(y,num_classes):
    new_ys = []
    for i in range(len(y)):
        new_y = []
        for j in range(len(y[i])):
            id = y[i][j]
            l = np.zeros((num_classes))
            if id != 0:
                l[id - 1] = 1
            new_y.append(l)
        new_ys.append(new_y)
    return np.array(new_ys)





def predict(logits,nums_step,lenghts):
        #logits.shape = [batch_size,nums_steps,nums_classes]
        #print logits.shape
    assert len(logits) == len(lenghts)
    y_preds = []
    for i in range(len(logits)):
        y_pred = []
        length = lenghts[i]
        for j in range(len(logits[i])):
            if j >= length:
                y_pred.append(0)
                continue
            id = np.where(logits[i][j] == np.max(logits[i][j]))
            y_pred.append(id[0][0]+1)
        y_preds.append(y_pred)
    y_preds = np.array(y_preds)

        #y_pred = [batch_size,nums_step]
    return y_preds
y = np.array([[1,2,1,2,2],[2,3,1,0,0]])
y = reshape_to_batchnumsteps_numclasses(y,3)
print y
lengths = [5,3]
print predict(y,5,lengths)

#batch 50
#hidden  60 90
#echo:20
#evalution on train data is loss_train2.206839,precison0.307692,recall0.266667,f10.285714
#result 0.613


