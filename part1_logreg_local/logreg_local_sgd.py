import tensorflow as tf
import numpy as np
import time
import h5py

print(" Data Loading ... \n")

label_train = np.load("train_l.npy").astype(np.float32)
# tfidf_documents_test=np.load("test.npy").astype(np.float32)
label_test = np.load("test_l.npy").astype(np.float32)
h5f1 = h5py.File('train.h5', 'r')
tfidf_train = h5f1['d1'][:]
h5f2 = h5py.File('test.h5', 'r')
tfidf_test = h5f2['d2'][:]

print('train data shape: {}'.format(tfidf_train.shape))
print('train label data shape: {}'.format(label_train.shape))

print('\n Defining the graph ... \n')
# Parameters
learning_rate = 0.02
n_epochs = 54
batch_size = 4000

X = tf.placeholder(tf.float32, shape=[None, tfidf_train.shape[1]], name='image')
Y = tf.placeholder(tf.float32, shape=[None, 50], name='label')

W = tf.get_variable(name='weights', shape=(tfidf_train.shape[1], 50), initializer=tf.random_normal_initializer())
b = tf.get_variable(name='bias', shape=50, initializer=tf.random_normal_initializer())

# weight = tf.reshape(tf.reduce_sum(y,0)/tf.reduce_sum(tf.reduce_sum(y,0)),[1,50])
# %%
# Construct model
Z_out = tf.add(tf.matmul(X, W), b)
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=Z_out, name='loss')
L = tf.reduce_mean(entropy)
L2_regularizer = tf.nn.l2_loss(W)
loss = tf.reduce_mean(L + 0.01 * L2_regularizer)
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = opt.minimize(loss)
predicted_labels = tf.nn.softmax(Z_out)


# correct_preds= tf.equal(tf.argmax(preds,1), tf.argmax(Y,1))

def accuracy(predicted_labels, labels):
    print('inside accuracy function ... \n')
    p = 0
    for ix in range(predicted_labels.shape[0]):
        if labels[ix, np.argmax(predicted_labels[ix, :])] != 0:
            p += 1
    return 100.0 * p / labels.shape[0]


init = tf.global_variables_initializer()
writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())

with tf.Session() as sess:
    start = time.time()
    sess.run(init)
    n_batches = int(tfidf_train.shape[0] / batch_size)

    # train the model n_epochs times
    for i in range(n_epochs):
        total_loss = 0
        for j in range(n_batches):
            x, y = tfidf_train[i * batch_size:(i + 1) * batch_size, :], label_train[i * batch_size:(i + 1) * batch_size, :]
            _, loss_int = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += loss_int
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))
    print('Total time: {0} seconds'.format(time.time() - start))

    # test the model
    print(' Testing phase...')

    print('Train Accuracy', accuracy(sess.run(predicted_labels, feed_dict={X: tfidf_train}), label_train))
    print('Test Accuracy {0}'.format(accuracy(sess.run(predicted_labels, feed_dict={X: tfidf_test}), label_test)))
    # print('Accuracy {0}'.format(total_correct_preds/mnist_test_num_examples))
writer.close()
