import util
import time
import tensorflow as tf
import evaluation
from BILSTM import BILSTM
train_path ='restaurant2015/traindata'
test_path = 'restaurant2015/testdata'
model_path = 'lstm_model/save_net_gridsearch.ckpt'
emb_path = 'data/vectors-150.txt'
emb_dim = 150
eval_path='restaurant2015/evaluation_data'
output_path='restaurant2015/result'
print 'grid search begin!'
def grid_search():
    num_layers = [1,2]
    num_hiddens = [100,150,200]
    num_steps = [30,50,70]
    for layer in num_layers:
        for hidden in num_hiddens:
            for step in num_steps:
                print 'paras:layer:%d,hiddens:%d,step:%d'%(layer,hidden,step)
                train(layer,hidden,step)
                test(layer,hidden,step)

def train(num_layers,num_hiddens,num_steps):
    X_eval = None
    y_eval = None
    X_train, y_train,X_eval,y_eval= util.getTrainEvalData(train_path=train_path,
                                                         seq_max_len=num_steps)
    word2id, _ = util.loadMap('data/word2id')
    label2id, _ = util.loadMap('data/label2id')
    num_words = len(word2id)
    num_classes = len(label2id)
    if emb_path != None:
        embedding_matrix = util.getEmbedding(emb_path)
    else:
        embedding_matrix = None
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-0.2, 0.2)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = BILSTM(
                num_words,
                num_classes,
                num_hiddens=num_hiddens,
                num_steps=num_steps,
                num_layers=num_layers,
                embedding_matrix=embedding_matrix,
                emb_dim=emb_dim)
        # self,num_hiddens,num_steps,embedding_matrix,num_words,num_classes,is_training=True

        init = tf.initialize_all_variables()
        sess.run(init)
        model.train(sess, model_path, X_train, y_train, X_eval, y_eval)

def test(num_layers,num_hiddens,num_steps):
    word2id, _ = util.loadMap('data/word2id')
    label2id, _ = util.loadMap('data/label2id')
    num_words = len(word2id)
    num_classes = len(label2id)
    if emb_path != None:
        embedding_matrix = util.getEmbedding(emb_path)
    else:
        embedding_matrix = None
    X_test, X_test_str, X_test_label_str = util.getTestData(test_path, num_steps, False)
    util.save_test_data(X_test_str, X_test_label_str, output_path=eval_path)
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-0.2, 0.2)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
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
        saver.restore(sess, model_path)
        model.test(sess, X_test, X_test_str, output_path=output_path)
        end_time = time.time()
    evaluation.evalution(gold_path=eval_path, pred_path=output_path)
if __name__ =='__main__':
    grid_search()