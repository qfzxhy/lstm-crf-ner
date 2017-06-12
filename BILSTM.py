import tensorflow as tf
# from tensorflow.models.rnn import rnn
import math
import numpy as np
import util
class BILSTM(object):
    def show_paras(self):
        print 'layers:%d'%self.num_layers
        print 'hiddens:%d'%self.num_hiddens
        print 'steps:%d'%self.num_steps
        print 'learing_rate:%f'%self.learning_rate
        print 'batchsize:%d'%self.batch_size
        print 'num_epochs%d'%self.num_epochs
        print 'l2_loss:%s'%self.if_l2
    def __init__(self,
                 num_words,
                 num_classes,
                 num_hiddens=1,
                 num_steps=1,
                 num_layers=1,
                 learning_rate=0.01,
                 num_epochs=20,
                 is_training=True,
                 embedding_matrix=None,
                 emb_dim=100,
                 is_crf=True,
                 l2_loss=False):
        # self.num_inputs = num_inputs
        #

        self.num_steps = num_steps
        self.num_words = num_words
        self.num_classess = num_classes
        self.emb_dim = emb_dim
        self.num_hiddens = num_hiddens
        # self.save_path = save_path
        self.num_layers = num_layers

        self.is_crf = is_crf
        self.if_l2 = l2_loss
        self.learning_rate = 0.05
        self.batch_size = 100
        self.num_epochs = 20
        self.max_f1 = -1.0

        self.show_paras()

        self.inputs = tf.placeholder(tf.int32,[None,self.num_steps])
        if not self.is_crf:
            self.targets = tf.placeholder(tf.int32,[None,self.num_steps,self.num_classess])
        else:
            self.targets = tf.placeholder(tf.int32,[None,self.num_steps])

        if embedding_matrix is not None:
            self.embedding = tf.Variable(embedding_matrix,trainable=True,name = 'emb',dtype=tf.float32)
        else:
            self.embedding = tf.get_variable('emb',[self.num_words,self.emb_dim])

        self.inputs_emb = tf.nn.embedding_lookup(self.embedding,self.inputs)
        if is_training:
            self.inputs_emb = tf.nn.dropout(self.inputs_emb,0.8)
        #length = [batchsize]
        self.length = tf.reduce_sum(tf.sign(self.inputs),1)
        self.length = tf.cast(self.length,tf.int32)
        length_64 = tf.cast(self.length, tf.int64)
        forward_output, _ = tf.nn.dynamic_rnn(
            tf.contrib.rnn.LSTMCell(self.num_hiddens),
            self.inputs_emb,
            dtype=tf.float32,
            sequence_length=self.length,
            scope="RNN_forward")
        backward_output_, _ = tf.nn.dynamic_rnn(
            tf.contrib.rnn.LSTMCell(self.num_hiddens),
            inputs=tf.reverse_sequence(self.inputs_emb,
                                       length_64,
                                       seq_dim=1),
            dtype=tf.float32,
            sequence_length=self.length,
            scope="RNN_backword")

        backward_output = tf.reverse_sequence(backward_output_,
                                          length_64,
                                          seq_dim=1)

        output = tf.concat([forward_output, backward_output], 2)
        output = tf.reshape(output, [-1, self.num_hiddens * 2])
        # if is_training:
        #     output = tf.nn.dropout(output, 0.5)

        self.softmax_w = tf.get_variable('softmax_w', [2 * self.num_hiddens, self.num_classess])
        self.softmax_b = tf.get_variable('softmax_b', [self.num_classess])
        matricized_unary_scores = tf.matmul(output, self.softmax_w) + self.softmax_b
        #unary_score = [batch_size,num_steps,num_classes]
        self.unary_scores = tf.reshape(
            matricized_unary_scores,
            [-1, self.num_steps, self.num_classess])#
        if not self.is_crf:
            self.loss_ = tf.nn.softmax_cross_entropy_with_logits(logits=self.unary_scores,labels=self.targets)
            self.loss = tf.reduce_mean(self.loss_)
            if l2_loss:
                l2_reg = tf.nn.l2_loss(self.softmax_w) + tf.nn.l2_loss(self.softmax_b)
                self.loss += 0.001 * l2_reg
        else:

            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.unary_scores, self.targets, self.length)
            self.loss = tf.reduce_mean(-self.log_likelihood)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self,sess,save_path,X_train,y_train,X_eval,y_val):

        #X_train = [all_sentence_size,num_steps]
        #y_train = [all_sentence_size,num_steps]
        _,id2word = util.loadMap('data/word2id')
        _,id2label = util.loadMap('data/label2id')
        saver = tf.train.Saver()
        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))
        # print y_train.shape
        for epoch in range(self.num_epochs):
            sh_index = np.arange(len(X_train))
            np.random.shuffle(sh_index)

            X_train = X_train[sh_index]
            y_train = y_train[sh_index]

            # print y_train.shape
            print "current epoch: %d" % (epoch)
            for iteration in range(num_iterations):
                # X_train_batch = [batch_size,num_steps]
                # y_train_batch = [batch_size,num_steps]
                X_train_batch,y_train_batch = util.nextBatch(X_train,y_train,start_index = iteration*self.batch_size,batch_size = self.batch_size)


                if not self.is_crf:
                    # y_train_batch = [batchsize,numsteps,numclasses]
                    y_train_batch = util.reshape_to_batchnumsteps_numclasses(y_train_batch, self.num_classess)

                feed_dict = {self.inputs: X_train_batch,
                             self.targets: y_train_batch}
                # \
                if not self.is_crf:
                    _,unary_scores,lengths,loss_train\
                        = sess.run([
                        self.optimizer,
                        self.unary_scores,
                        self.length,
                        self.loss
                        ],
                             feed_dict = feed_dict)
                    if iteration % 10 == 0:
                        y_batch_pred = self.predict(unary_scores, self.num_steps)
                        # y_batch_pred = self.nn_decode_batch(sess,unary_scores,lengths)
                        y_train_batch = util.reshape_to_batch_numsteps(y_train_batch, self.num_classess, self.num_steps)
                        precison, recall, f1 = self.evaluate(X_train_batch, y_train_batch, y_batch_pred, lengths,
                                                             id2word=id2word, id2label=id2label)
                        print 'evalution on train data is loss_train%f,precison%f,recall%f,f1%f' % (
                        loss_train, precison, recall, f1)

                else:
                    _,unary_scores,lengths,loss_train,transMatrix\
                        = sess.run([
                        self.optimizer,
                        self.unary_scores,
                        self.length,
                        self.loss,
                        self.transition_params
                        ],
                             feed_dict = feed_dict)
                    if iteration % 10 == 0:
                        y_batch_pred_crf = self.viterbi_decode_batch(unary_scores,lengths,transMatrix)
                        precison,recall,f1 = self.evaluate(X_train_batch,y_train_batch,y_batch_pred_crf,lengths,id2word=id2word,id2label=id2label)
                        print 'evalution on train data is loss_train%f,precison%f,recall%f,f1%f'%(loss_train,precison,recall,f1)

                if iteration % 10 == 0:
                    # y_eval_pred = self.decode_all(sess,X_eval)
                    #y_val = [-1,num_steps]
                    #y_eval = [-1,num_steps,num_classes]
                    y_eval_pred,eval_loss,eval_seq_len = self.decode_all(sess,X_eval,y_val)
                    precison, recall, f1 = self.evaluate(X_eval, y_val, y_eval_pred, eval_seq_len, id2word=id2word,
                                                             id2label=id2label)
                    print 'evalution on eval data is eval_loss%f,precison%f,recall%f,f1%f' % (
                        eval_loss, precison, recall, f1)

                    if f1 > self.max_f1:
                        self.max_f1 = f1
                        saver.save(sess,save_path)
                        print 'the best model evaltion:f1:%f'%self.max_f1

    def decode_all(self,sess,Xtest,ytest):
        all_y_pred = []
        all_loss = []
        all_test_seq_len = np.zeros(len(Xtest))
        num_iterations = int(math.ceil(1.0 * len(Xtest) / self.batch_size))
        for itor in range(num_iterations):
            start_index = itor * self.batch_size
            end_index = min(start_index + self.batch_size, len(Xtest))
            X_test_batch = Xtest[start_index:end_index]
            y_test_batch = ytest[start_index:end_index]

            if not self.is_crf:
                # unary_score = [basize,num_stepes,num_classees]
                # test_seq_len = [batch_size]
                y_test_batch = util.reshape_to_batchnumsteps_numclasses(y_test_batch, self.num_classess)
                feed_dict = {self.inputs: X_test_batch,
                             self.targets: y_test_batch}
                unary_score, test_seq_len,loss = sess.run(
                    [self.unary_scores,
                     self.length,
                     self.loss],
                    feed_dict=feed_dict
                )
                all_y_pred.extend(self.nn_decode_batch(unary_score,test_seq_len))
                all_loss.append(loss)
                all_test_seq_len[start_index:end_index] = test_seq_len
            else:
                feed_dict = {self.inputs: X_test_batch,
                             self.targets: y_test_batch}
                # unary_score = [basize,num_stepes,num_classees]
                # test_seq_len = [batch_size]
                unary_score, test_seq_len,transMatrix,loss = sess.run(
                    [self.unary_scores,
                     self.length,
                     self.transition_params,
                     self.loss],
                    feed_dict=feed_dict
                )
                all_y_pred.extend(self.viterbi_decode_batch(unary_score,test_seq_len,transMatrix))
                all_loss.append(loss)
                all_test_seq_len[start_index:end_index] = test_seq_len
        return all_y_pred,np.mean(np.array(all_loss)),all_test_seq_len

    def nn_decode_batch(self,unary_scores,test_seq_len):
        # unary_scores = [batch_size,num_steps,num_classes]
        #return  list: [batch_size,seq_len]

        y_preds = []
        for tf_unary_scores_,seq_len_ in zip(unary_scores,test_seq_len):
            tf_unary_scores_ = tf_unary_scores_[:seq_len_]
            y_pred = []
            for j in range(len(tf_unary_scores_)):
                id = np.where(tf_unary_scores_[j] == np.max(tf_unary_scores_[j]))
                y_pred.append(id[0][0] + 1)
            y_preds.append(y_pred)
        return y_preds

    def viterbi_decode_batch(self,unary_scores,test_seq_len,transMatrix):
        #unary_scores = [batch_size,num_steps,num_classes]
        #return  list: [batch_size,seq_len]


        y_pred = []

        for tf_unary_scores_,seq_len_ in zip(unary_scores,test_seq_len):
            tf_unary_scores_ = tf_unary_scores_[:seq_len_]
            #viterbi_sequence = [num_steps]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, transMatrix)
            y_pred.append(viterbi_sequence)
                # y_gold.append(y_)
        return y_pred

        pass
    def test(self,sess,X_test,X_test_str,output_path):
        precision = []
        recall = []
        label2id,id2label = util.loadMap('data/label2id')
        num_iterations = int(math.ceil(1.0 * len(X_test) / self.batch_size))
        # print "number of iteration:" + str(num_iterations)
        with open(output_path,'w') as outfiler:
            for iteration in range(num_iterations):
                # print 'iteration:' + str(iteration+1)
                start_index = iteration*self.batch_size
                end_index = min(start_index + self.batch_size,len(X_test))
                X_test_batch = X_test[start_index:end_index]
                X_test_batch_str = X_test_str[start_index:end_index]
                if not self.is_crf:
                    logits_test = sess.run(
                        [
                            self.unary_scores
                        ],
                        feed_dict =\
                        {
                            self.inputs:X_test_batch
                        }
                    )
                    y_pred = self.predict(logits_test[0],self.num_steps)
                else:
                    unary_scores,transMatrix,test_seq_len = sess.run(
                        [
                            self.unary_scores,
                            self.transition_params,
                            self.length
                        ],
                        feed_dict = \
                        {
                            self.inputs:X_test_batch
                        })
                    y_pred = self.viterbi_decode_batch(unary_scores,test_seq_len,transMatrix)

                for batch_id in range(len(X_test_batch)):
                    for word_id in range(len(X_test_batch_str[batch_id])):
                        outfiler.write(X_test_batch_str[batch_id][word_id] + '\t' + id2label[y_pred[batch_id][word_id]]+'\n')
                    outfiler.write('\n')

    def predict(self,logits, nums_step):
        # logits.shape = [batch_size,num_steps,num_classes]
        # return [batch_size,num_steps]
        y_preds = []
        for i in range(len(logits)):
            y_pred = []
            # length = lenghts[i]
            for j in range(len(logits[i])):
                # if j >= length:
                #     y_pred.append(0)
                #     continue
                id = np.where(logits[i][j] == np.max(logits[i][j]))
                y_pred.append(id[0][0] + 1)
            y_preds.append(y_pred)
        y_preds = np.array(y_preds)
        return y_preds
    # def predict(self,logits,nums_step):
    #     #logits.shape = [batch_size,nums_steps,nums_classes]
    #     #print logits.shape
    #     y_pred = []
    #     for logit in logits:
    #         id = np.where(logit == np.max(logit))
    #         y_pred.append(id[0][0]+1)
    #     y_pred = np.array(y_pred)
    #     y_pred = np.reshape(y_pred,[-1,nums_step])
    #     #y_pred = [batch_size,nums_step]
    #     return y_pred

    def evaluate(self,X,y_gold,y_pred,test_seq_len,id2word,id2label):
        #y_gold = list : [batch_size,seq_len]
        #y_pred = [batch_size,num_steps]
        precision = -1.0
        recall = -1.0
        f1 = -1.0
        num_pred = 0
        num_gold = 0
        num_right = 0
        for i in range(len(y_gold)):
            seq_len = int(test_seq_len[i])
            x = [id2word[X[i][j]] for j in range(seq_len)]
            y = ''.join([id2label[y_gold[i][j]] for j in range(seq_len)])
            y_p = ''.join([id2label[y_pred[i][j]] for j in range(seq_len)])
            entity_gold = util.extractEntity(x,y)
            entity_pred = util.extractEntity(x,y_p)
            num_right += len(set(entity_gold)&set(entity_pred))
            num_gold += len(set(entity_gold))
            num_pred += len(set(entity_pred))
        if num_pred != 0:
            precision = 1.0 * num_right / num_pred
        if num_gold != 0:
            recall = 1.0 * num_right / num_gold
        if precision>0 and recall>0:
            f1 = 2 * (precision * recall) / (recall + precision)
        return precision,recall,f1
