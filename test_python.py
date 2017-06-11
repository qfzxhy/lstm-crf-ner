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

#
# y_eval = util.reshape_to_batchnumsteps_numclasses(y_val, self.num_classess)
# loss_eval, logits_eval, test_seq_len = sess.run([
#     self.loss,
#     self.unary_scores,
#     self.length,
# ],
#     feed_dict={
#         self.inputs: X_eval,
#         self.targets: y_eval
#     })
# y_eval_pred = self.predict(logits_eval, self.num_steps)
# precison, recall, f1 = self.evaluate(X_eval, y_val, y_eval_pred, test_seq_len, id2word=id2word,
#                                      id2label=id2label)
# print 'evalution on eval data is eval_loss%f,precison%f,recall%f,f1%f' % (
#     loss_eval, precison, recall, f1)
# else:
# loss_eval, unary_score, test_seq_len, transMatrix = sess.run(
#     [self.loss,
#      self.unary_scores,
#      self.length,
#      self.transition_params],
#     feed_dict=feed_dict
# )
# print unary_scores.shape
# y_eval_pred_crf = self.viterbi_decode_batch(unary_scores, test_seq_len, transMatrix)
# print len(y_eval_pred_crf)
# print len(y_val)


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

def test(y_val,num_classess,num_steps):
    import util

    y_eval = util.reshape_to_batchnumsteps_numclasses(y_val, num_classess)
    print y_eval
    y_eval = util.reshape_to_batch_numsteps(y_eval, num_classess, num_steps)
    print y_eval
y = np.array([[1,2,1,2,2],[2,3,1,0,0]],dtype=int)
test(y,3,5)
# y = reshape_to_batchnumsteps_numclasses(y,3)
# print y
# lengths = [5,3]
# print predict(y,5,lengths)

#batch 50
#hidden  60 90
#echo:20
#evalution on train data is loss_train2.206839,precison0.307692,recall0.266667,f10.285714
#result 0.613


