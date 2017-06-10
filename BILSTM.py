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
                 is_crf=False,
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
        self.batch_size = 50
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

        self.inputs_emb = tf.nn.dropout(self.inputs_emb,0.8)

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
    # matricized_unary_scores = tf.nn.log_softmax(matricized_unary_scores)
        self.unary_scores = tf.reshape(
            matricized_unary_scores,
            [-1, self.num_steps, self.num_classess])  #
        # self.outputs,_,_ = rnn.bidirectional_rnn(
        #     lstm_fw_cell,
        #     lstm_bw_cell,
        #     self.inputs_emb,
        #     dtype=tf.float32,
        #     sequence_length=self.length
        # )
        #fw bw concat
        # self.outputs = tf.reshape(tf.concat(1, self.outputs), [-1, 2 * self.num_hiddens])

        # shape = [batch_size*num_steps,num_classes]
        # self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
        # self.logits = tf.nn.dropout(self.logits,0.8
        if not self.is_crf:
            self.loss_ = tf.nn.softmax_cross_entropy_with_logits(logits=self.unary_scores,labels=self.targets)
            self.loss = tf.reduce_mean(self.loss_)
            if l2_loss:
                l2_reg = tf.nn.l2_loss(self.softmax_w) + tf.nn.l2_loss(self.softmax_b)
                self.loss += 0.001 * l2_reg
        else:

            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.unary_scores, self.targets, self.length)
            self.loss = tf.reduce_mean(-log_likelihood)

                 # else:
        #     #crf
        #     self.tag_scores = tf.reshape(self.logits,[self.batch_size,self.num_steps,self.num_classess])
        #     self.transitions = tf.get_variable("transitions",[self.num_classess+1,self.num_classess+1])
        #     dummy_val = -1000
        #     class_pad = tf.Variable(dummy_val*np.ones((self.batch_size,self.num_steps,1)),dtype=tf.float32)
        #     self.observations = tf.concat(2,[self.tag_scores,class_pad])
        #     begin_vec = tf.Variable(np.array([[dummy_val]*self.num_classess + [0] for _ in range(self.batch_size)]),trainable=False,dtype=tf.float32)
        #     end_vec = tf.Variable(np.array([[0] + [dummy_val]*self.num_classess for _ in range(self.batch_size)]),trainable=False,dtype=tf.float32)
        #     begin_vec = tf.reshape()
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def get_crf_loss(self):
        pass

    def train(self,sess,save_path,X_train,y_train,X_val,y_val):

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

                _,loss_train,logits_train,lengths = sess.run([
                    self.optimizer,
                    self.loss,
                    self.unary_scores,
                    self.length
                    ],
                         feed_dict = {
                             self.inputs:X_train_batch,
                             self.targets:y_train_batch,
                         })
                if iteration % 10 == 0:
                    # logits_train = [batch_size, num_steps, num_classes]
                    if not self.is_crf:
                        #y_pred = [batch_size, num_steps]
                        y_pred = self.predict(logits_train,self.num_steps)
                        #y_train_batch = [batch_size,num_steps]
                        y_train_batch = util.reshape_to_batch_numsteps(y_train_batch,self.num_classess,self.num_steps)
                    # else:
                    #     y_pred =
                    precison,recall,f1 = self.evaluate(X_train_batch,y_train_batch,y_pred,id2word=id2word,id2label=id2label)
                    print 'evalution on train data is loss_train%f,precison%f,recall%f,f1%f'%(loss_train,precison,recall,f1)
        # saver.save(sess,save_path)
       # saver.save(sess,save_path)


                if iteration % 10 == 0:
                    # X_eval_batch,y_eval_batch = util.nextRandomBatch(X_val,y_val,self.batch_size)
                    y_eval = util.reshape_to_batchnumsteps_numclasses(y_val,self.num_classess)
                    loss_eval, logits_eval = sess.run([
                        self.loss,
                        self.unary_scores
                        ],
                        feed_dict={
                            self.inputs: X_val,
                            self.targets: y_eval
                        })
                    y_pred = self.predict(logits_eval, self.num_steps)


                    # y_pred = util.reshape_to_batch_numsteps(y_pred,self.num_classess,self.num_steps)
                    y_eval = util.reshape_to_batch_numsteps(y_eval, self.num_classess, self.num_steps)
                    # print y_eval_batch
                    # print y_pred
                    precison, recall, f1 = self.evaluate(X_val, y_eval, y_pred, id2word=id2word,
                                                         id2label=id2label)
                    print 'evalution on eval data is eval_loss%f,precison%f,recall%f,f1%f' % (loss_eval,precison, recall, f1)
                    if f1 > self.max_f1:
                        self.max_f1 = f1
                        saver.save(sess,save_path)
                        print 'the best model evaltion:f1:%f'%self.max_f1

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
                X_test_batch = [X_test[i] for i in range(start_index,end_index)]
                X_test_str_batch = [X_test_str[i] for i in range(start_index,end_index)]
                X_test_batch = np.array(X_test_batch)
                logits_test = sess.run(
                    [
                        self.unary_scores
                    ],
                    feed_dict =\
                    {
                        self.inputs:X_test_batch
                    }
                )
                # print logits_test
                y_pred = self.predict(logits_test[0],self.num_steps)

                for batch_id in range(len(X_test_batch)):
                    for word_id in range(len(X_test_str_batch[batch_id])):
                        outfiler.write(X_test_str_batch[batch_id][word_id] + '\t' + id2label[y_pred[batch_id][word_id]]+'\n')
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

    def evaluate(self,X,y_gold,y_pred,id2word,id2label):
        #y_gold = [batch_size,num_steps]
        #y_pred = [batch_size,num_steps]
        precision = -1.0
        recall = -1.0
        f1 = -1.0
        num_pred = 0
        num_gold = 0
        num_right = 0
        for i in range(len(y_gold)):
            x = [id2word[val] for val in X[i] if val != 0]
            y = ''.join([id2label[val] for val in y_gold[i] if val != 0])
            y_p = ''.join([id2label[val] for val in y_pred[i] if val != 0])
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
