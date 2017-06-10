import util
import time
import tensorflow as tf
from BILSTM import BILSTM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('train_path',help = "the path of the train file",)
parser.add_argument('save_path',help = 'the path of the save model path',)
parser.add_argument('-v','--val_path',help = 'the path of the validation file',default=None)
parser.add_argument('-e','--epoch',help = 'the number of epoch',default=15,type=int)
parser.add_argument('-w','--wordemb_path',help = 'the path of the word embedding file',default=None)
parser.add_argument('-d','--wordemb_dim',help='the embedding dim',default=100)

parser.add_argument('--num_steps',default=30)
parser.add_argument('--num_hiddens',default=28)
parser.add_argument('--num_layers',default=1)

args = parser.parse_args()
train_path = args.train_path
save_path = args.save_path

# train_path = 'restaurant2015/traindata'
eval_path = args.val_path
emb_path = args.wordemb_path
num_epochs = args.epoch
emb_dim = int(args.wordemb_dim)

num_steps = int(args.num_steps)
num_hiddens = int(args.num_hiddens)
num_layers = int(args.num_layers)
start_time = time.time()
X_eval = None
y_eval = None
print "prepare train and validation data"
#X_train,y_train = util.getTrainData(train_path,seq_max_len=num_steps)

X_train,y_train,X_eval,y_eval = util.getTrainEvalData(train_path = train_path, seq_max_len = num_steps)
print 'X_train.shape:'+str(X_train.shape)
print 'y_train.shape:'+str(y_train.shape)
if X_eval is not None:
    print 'X_eval.shape:'+str(X_eval.shape)

word2id,_ = util.loadMap('data/word2id')
label2id,_ = util.loadMap('data/label2id')
num_words = len(word2id)
num_classes = len(label2id)
if emb_path != None:
    embedding_matrix = util.getEmbedding(emb_path)
else:
    embedding_matrix = None
# print y_eval
with tf.Session() as sess:
    initializer = tf.random_uniform_initializer(-0.1,0.1)
    with tf.variable_scope("model",reuse=None,initializer = initializer):
          model = BILSTM(
              num_words,
              num_classes,
              num_hiddens = num_hiddens,
              num_steps=num_steps,
              num_layers=num_layers,
              embedding_matrix=embedding_matrix,
              emb_dim=emb_dim)
    #self,num_hiddens,num_steps,embedding_matrix,num_words,num_classes,is_training=True

    init = tf.global_variables_initializer()
    sess.run(init)
#    sess.run(model.logits)
#     print y_train.shape
    #sess.run(model.inputs)
    # X_train_batch, y_train_batch = util.nextBatch(X_train, y_train, start_index=0,
    #                                               batch_size=100)
    #
    # y_train_batch = util.reshape_to_batchnumsteps_numclasses(y_train_batch, 3)
    # # y_train = util.reshape_to_batchnumsteps_numclasses(y_train, 3)
    # # print X_train_batch.shape
    # # print y_train_batch.shape
    # # print sess.run([],feed_dict={model.inputs:X_train_batch,model.targets:y_train_batch})
    #
    # input_emb,outputs,loss_,opt, loss_train, logits_train = sess.run([
    #     model.inputs_emb,
    #     model.outputs,
    #     model.loss_,
    #     model.optimizer,
    #     model.loss,
    #     model.logits
    #
    # ],
    #     feed_dict={
    #         model.inputs: X_train_batch,
    #         model.targets: y_train_batch
    #     })
    model.train(sess,save_path,X_train,y_train,X_eval,y_eval)
    # print input_emb.shape
    # print outputs.shape
    # print loss_.shape
    # print loss_train.shape
    # print logits_train.shape
    end_time = time.time()
    print 'time:%f(hour)' % ((end_time - start_time)/3600)