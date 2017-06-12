import time
import util
import argparse
import tensorflow as tf
from BILSTM import BILSTM
import evaluation



start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('test_path',help='the path of test file')
parser.add_argument('model_path',help='the save model path')
parser.add_argument('output',help='the result path')
parser.add_argument('-w','--wordemb_path',help = 'the path of the word embedding file',default=None)
parser.add_argument('-d','--wordemb_dim',help='the embedding dim',default=100)

parser.add_argument('--num_steps',default=30)
parser.add_argument('--num_hiddens',default=28)
parser.add_argument('--num_layers',default=1)

args = parser.parse_args()
model_path = args.model_path
test_path = args.test_path
emb_path = args.wordemb_path
emb_dim = int(args.wordemb_dim)
output_path = args.output
evaluation_path = 'evaluation_data'
num_steps = int(args.num_steps)
num_hiddens = int(args.num_hiddens)
num_layers = int(args.num_layers)
is_crf =True
print 'paras:'
print 'num_steps:%d'%num_steps
print 'num_hiddens:%d'%num_hiddens
print 'num_layers:%d'%num_layers
print 'wordemb:%s'%emb_path
word2id,_ = util.loadMap('data/word2id')
label2id,_ = util.loadMap('data/label2id')
num_words = len(word2id)
num_classes = len(label2id)
if not is_crf:
    num_classes = num_classes-2
if emb_path != None:
    embedding_matrix = util.getEmbedding(emb_path)
else:
    embedding_matrix = None
X_test,X_test_str,X_test_label_str = util.getTestData(test_path,num_steps,False)
# print X_test_label_str
util.save_test_data(X_test_str,X_test_label_str,output_path=evaluation_path)
with tf.Session() as sess:
    initializer = tf.random_uniform_initializer(-0.1,0.1)
    with tf.variable_scope("model",reuse=None,initializer = initializer):
        model = BILSTM(
            num_words,
            num_classes,
            num_hiddens=num_hiddens,
            num_steps=num_steps,
            num_layers=num_layers,
            embedding_matrix=embedding_matrix,
            emb_dim=emb_dim)
    print 'loading lstm model parameters!'
    saver = tf.train.Saver()
    saver.restore(sess,model_path)
    model.test(sess,X_test,X_test_str,output_path=output_path)
    end_time = time.time()
    print 'test time is %f(hour)'%((end_time-start_time)/3600)
evaluation.evalution(gold_path=evaluation_path,pred_path=output_path)