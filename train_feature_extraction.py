import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import numpy as np
import time

n_classes = 43
EPOCHS = 1
BATCH_SIZE = 128

# TODO: Load traffic signs data.

with open("train.p", mode='rb') as f:
    train = pickle.load(f)
X_train, y_train = train['features'], train['labels']

# TODO: Split data into training and validation sets.
n_train = int(((X_train.shape[0]+1)/30)*0.9)*30
n_validation = X_train.shape[0]-n_train

# # Total tracks = Total images/30 where 30 is the number of frames per second
# tot_tracks = int(X_train.shape[0]/30)

# # rand_train stores random values that will later be used to select random tracks from the training set
# rand_train = np.random.choice(np.arange(0,tot_tracks),replace=False,size=int((n_train+n_validation)/30))

# # Initialize simple arrays for datasets
# X_training = []
# y_training = []
# X_validation = []
# y_validation = []

# # Split training dataset by randomly distributing its "tracks" into training and validation  
# for x in range(len(rand_train)):
#     for y in range(30):
#         if(x>=n_validation/30):
#             X_training.append(X_train[(rand_train[x]*30)+y])
#             y_training.append(y_train[(rand_train[x]*30)+y])
#         else:
#             X_validation.append(X_train[(rand_train[x]*30)+y])
#             y_validation.append(y_train[(rand_train[x]*30)+y])

# # Final packing of the training and validation datasets
# X_training = np.array(X_training)
# X_training = X_training[:5000]
# y_training = np.array(y_training,dtype=np.int32)
# y_training = y_training[:5000]
# X_validation = np.array(X_validation)
# y_validation = np.array(y_validation,dtype=np.int32)

X_training, X_validation, y_training, y_validation = train_test_split(X_train, y_train, test_size=0.33, random_state=30)
X_training = X_training[:5000]
y_training = y_training[:5000]
print(X_training.shape)
# TODO: Define placeholders and resize operation.

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))
y = tf.placeholder(tf.int64, None)
# y = tf.one_hot(y, n_classes)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.

shape = (fc7.get_shape().as_list()[-1], n_classes)  # use this shape for the weight matrix
fc8W = tf.Variable(tf.truncated_normal(shape=shape,dtype = tf.float32))
fc8b = tf.Variable(tf.zeros(n_classes))
logits = tf.nn.xw_plus_b(fc7,fc8W,fc8b)
# logits = tf.nn.softmax(result)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)
loss_operation = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])

pred = tf.arg_max(logits, 1)
acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))

# TODO: Train and evaluate the feature extraction model.

# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
# accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# cross_ent = tf.nn.softmax_cross_entropy_with_logits(logits, y)
# loss_opera = tf.reduce_mean(cross_ent)
# print(conv1_W)
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy,loss = sess.run([acc,loss_operation], feed_dict={x: batch_x, y: batch_y})
        total_loss += (loss * len(batch_x))
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples , total_loss/num_examples

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    num_examples = len(X_training)
    val_arr = np.zeros((EPOCHS+1))
    X_validation, y_validation = shuffle(X_validation, y_validation)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        start = time.time()
        X_training, y_training = shuffle(X_training, y_training)
        X_validation, y_validation = shuffle(X_validation, y_validation)

        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_training[offset:end], y_training[offset:end]
            op = sess.run([training_operation], 
                                  feed_dict={x: batch_x, y: batch_y})
            
        training_accuracy, loss_train = evaluate(X_training,y_training)#X_train[:,:,:,0:1],y_train)
        validation_accuracy, loss_val = evaluate(X_validation, y_validation)

        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f} \tTraining Loss = {}".format(training_accuracy,loss_train))
        print("Validation Accuracy = {:.3f} \tValidation Loss = {}".format(validation_accuracy,loss_val))
        print("Time for this EPOCH = ",time.time()-start)

        val_arr[i] = validation_accuracy
    print("Training completed")



