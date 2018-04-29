from __future__ import print_function,division
import tensorflow as tf
from get_bottleneck_values import get_bottleneck_values,get_bottleneck_list,iter_dataset
#import matplotlib.pyplot as plt
import os
import pickle

'''
Class for the convolutional neural network
'''
class CNNModel(object):

    def __init__(self,batch_size):
        self.file_path='./data/experiment1/'
        self.batch_size=batch_size
        self.train_epoch=50
        self.class_count=50
        self.sess=tf.Session()

    '''
    Function to build the computational graph
    Just a fully connected layer
    '''
    def build(self):

        self.bottleneck_input = tf.placeholder(tf.float32, shape=[None, 2048])
        self.y_=tf.placeholder(tf.float32, shape=[None, self.class_count])
        self.regularizer=tf.contrib.layers.l2_regularizer(scale=1e-4)
        self.drop_prob_dense = tf.placeholder(tf.float32)
        self.IsTraining = tf.placeholder(tf.bool)

        # just a fully connected layer


        logits = tf.layers.dense(inputs=self.bottleneck_input, units=self.class_count, activation=tf.nn.relu,
                                 kernel_regularizer =self.regularizer)


        self.y = tf.nn.softmax(logits)

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))

        #train_step = tf.train.AdamOptimizer(7e-4).minimize(cross_entropy)
        self.y_p = tf.argmax(self.y, 1)
        self.y_t = tf.argmax(self.y_, 1)
        #calculate the accuracy
        acc, acc_op = tf.metrics.accuracy(labels=self.y_t, predictions=self.y_p)


        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr=1e-3
        self.lr_decay_step=400
        self.lr_decay_rate=0.95

        self.learning_rate = tf.train.exponential_decay(
            self.lr,  # Base learning rate.
            self.global_step,  # Current index into the dataset.
            self.lr_decay_step,  # Decay step.
            self.lr_decay_rate,  # Decay rate.
            staircase=True
        )

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            try:
                self.train_step = optimizer.minimize(self.cross_entropy, global_step=self.global_step)
            except Exception as e:
                print(e)
    '''
    Function to train the model
    '''

    def train(self):

        self.build()

        #with tf.Session() as sess:
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.iteration_error=[]
        for epoch_i in range(self.train_epoch):
            step_error=0
            batch_num=1
            for batch_i in iter_dataset(file_path=self.file_path,model='train',batch_size=self.batch_size):
                self.train_step.run(session=self.sess,feed_dict={self.x: batch_i[0],self.y_: batch_i[1], self.IsTraining:True})
                ce = self.cross_entropy.eval(session=self.sess,feed_dict={self.x: batch_i[0],self.y_: batch_i[1],self.IsTraining:True})
                step_error+=ce

                if batch_num%10==0:
                    print('Epoch %d, batch %d, cross_entropy %g' % (epoch_i+1,batch_num, ce))
                batch_num+=1
            self.iteration_error.append(step_error)
        print('Cross Entropy Change:',self.iteration_error)
        #get training accuracy
        y_pred_training=[]
        y_true_training=[]
        for batch_i in iter_dataset(file_path=self.file_path,model='train',batch_size=self.batch_size):
            y_pred_training+=list(self.y_p.eval(session=self.sess,feed_dict={self.x: batch_i[0],self.y_: batch_i[1], self.IsTraining:True}))
            y_true_training+=list(self.y_t.eval(session=self.sess,feed_dict={self.x: batch_i[0],self.y_: batch_i[1], self.IsTraining:True}))
        #calculate accuracy
        p_correct=0
        for ii in range(len(y_pred_training)):
            if y_pred_training[ii]==y_true_training[ii]:
                p_correct+=1
        train_acc=p_correct/len(y_pred_training)
        print('Training Accuracy:',train_acc)

        #store the loss change
        with open('Loss_change_list.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.iteration_error, f, pickle.HIGHEST_PROTOCOL)

    '''
    Function to test the model on the test dataset
    '''
    def test(self):
        y_prediction=[]
        y_true=[]
        for batch_i in iter_dataset(file_path=self.file_path,model='test',batch_size=self.batch_size):
            #self.train_step.run(session=self.sess,feed_dict={self.x: batch_i[0],self.y_: batch_i[1], self.drop_prob: 0.5,self.IsTraining:False,self.drop_prob_dense:0.2})
            y_prediction += list(self.y_p.eval(session=self.sess,feed_dict={self.x: batch_i[0],self.y_: batch_i[1],self.IsTraining:False}))
            y_true += list(self.y_t.eval(session=self.sess,feed_dict={self.x: batch_i[0],self.y_: batch_i[1],self.IsTraining:False}))
        #print('Prediction:',y_prediction)
        #print('True:',y_true)
        #calculate accuracy
        p_correct=0
        for ii in range(len(y_prediction)):
            if y_prediction[ii]==y_true[ii]:
                p_correct+=1
        acc=p_correct/len(y_prediction)
        print('Accuracy:',acc)

    '''
    Function to show the figure the loss change
    '''
    '''
    def show_loss_change(self):
        plt.figure()
        x_axle=[(a+1) for a in range(len(self.iteration_error))]
        plt.plot(x_axle, self.iteration_error,linewidth=2)
        plt.title('Loss change of RGB images ', fontsize=20)
        plt.xlabel('Epoch Time', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.show()
    '''


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
    Model=CNNModel(20)
    Model.train()
    Model.test()
