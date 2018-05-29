import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_train = pd.read_csv('./input/train.csv')
data_test = pd.read_csv('./input/test.csv')
ids = data_test['PassengerId']

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # reads in the MNIST dataset


#Set up a linear classifier
classifier = tf.estimator.LinearClassifier
nonNumericColMap = {}

def handle_non_numeric_data(data, colMap):
    columns = data.columns
    for column in columns:
        if column not in colMap:
            colMap[column] = {}
        text_digit_vals = colMap[column]
        if data[column].dtype != np.int64 and data[column].dtype != np.float64:
            column_contexts = data[column].values.tolist()
            unique_elements = set(column_contexts)
            num_seen = 0
            for elem in unique_elements:
                if elem not in text_digit_vals:
                    text_digit_vals[elem] = num_seen
                    num_seen+=1
            data[column] = list(map(lambda x : text_digit_vals[x], data[column]))
    return data

def remove_nans(data):
    columns = data.columns
    for column in columns:
        data[column] = data[column].fillna(-1)
    return data

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    
    
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df, colMap):
    df = format_name(df)
    df = handle_non_numeric_data(df, colMap)
    df = remove_nans(df)
    return df

#print(data_train.head())
#print("---------")
data_train = transform_features(data_train, nonNumericColMap)
data_test = transform_features(data_test, nonNumericColMap)
#print(data_train.head())

from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = data_test.columns #['Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
data_train, data_test = encode_features(data_train, data_test)
#print(data_train.head())




#----------------------------------------
# Start neural network here: http://rohanvarma.me/Neural-Net-Tensorflow/
#----------------------------------------


# some functions for quick variable creation
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape = shape))

# hyperparameters we will use
learning_rate = 0.1
hidden_layer_neurons = 50
num_iterations = 5000
number_of_inputs = len(data_test.columns)
number_of_outputs = 2
# placeholder variables
x = tf.placeholder(tf.float32, shape = [None, number_of_inputs]) # none = the size of that dimension doesn't matter. why is that okay here? 
y_ = tf.placeholder(tf.float32, shape = [None, number_of_outputs]) # number of output states. In binary classification that's 1

# create our weights and biases for our first hidden layer
W_1, b_1 = weight_variable([number_of_inputs, hidden_layer_neurons]), bias_variable([hidden_layer_neurons])

# compute activations of the hidden layer
h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

W_2_hidden = weight_variable([hidden_layer_neurons, 30])
b_2_hidden = bias_variable([30])
h_2 = tf.nn.relu(tf.matmul(h_1, W_2_hidden) + b_2_hidden)
# create our weights and biases for our output layer
W_2, b_2 = weight_variable([30, number_of_outputs]), bias_variable([number_of_outputs])
# compute the of the output layer
y = tf.matmul(h_2,W_2) + b_2




# define our loss function as the cross entropy loss
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))

# create an optimizer to minimize our cross entropy loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)

# functions that allow us to gauge accuracy of our model
correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # creates a vector where each element is T or F, denoting whether our prediction was right
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) # maps the boolean values to 1.0 or 0.0 and calculates the accuracy

# we will need to run this in our session to initialize our weights and biases. 
init = tf.global_variables_initializer()
import random
def genFakeLabels(num_rows, num_cols):
    output = np.zeros((num_rows, num_cols))
    for row in range(num_rows):
        output[row][int(random.random() * num_cols)] = 1
    return output

def normalizeData(data):
    mean_vector = np.mean(data, axis=0)
    variance_vector = np.var(data, axis=0)
    adjustRow = lambda x: (x - mean_vector) / variance_vector
    return np.apply_along_axis(adjustRow, 1, data), mean_vector, variance_vector

def generateLabels(labels):
    labels = np.array([labels]).T
    return np.apply_along_axis(lambda x: [1 if x[0] == 0 else 0, 0 if x[0] == 0 else 1], 1, labels)

from sklearn.model_selection import train_test_split
#fakeData = normalizeData(np.random.random((800, len(data_test.columns))))
#fakeResults = genFakeLabels(800, 2)

X_all, m_v, v_v = normalizeData(data_train.drop(['Survived'], axis=1))
y_all = generateLabels(data_train['Survived'])


num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test)

def getBatch(data_x, data_y, batch_num, batch_size):
    total_data = len(data_y)
    start_index = (batch_num * batch_size) % total_data
    cur_batch_size = batch_size if start_index + batch_size < total_data else total_data - start_index
    return data_x[start_index : start_index + cur_batch_size], data_y[start_index : start_index + cur_batch_size]


# launch a session to run our graph defined above. 
with tf.Session() as sess:
    sess.run(init) # initializes our variables
    for i in range(num_iterations):
        # get a sample of the dataset and run the optimizer, which calculates a forward pass and then runs the backpropagation algorithm to improve the weights
        batch = getBatch(X_train, y_train, i, 30)
        #print(batch[0])
        #print(batch[1])
        #break
        optimizer.run(feed_dict = {x: batch[0], y_: batch[1]})
        # every 100 iterations, print out the accuracy
        if i % 100 == 0:
            # accuracy and loss are both functions that take (x, y) pairs as input, and run a forward pass through the network to obtain a prediction, and then compares the prediction with the actual y.
            acc = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1]})
            loss = cross_entropy_loss.eval(feed_dict = {x: batch[0], y_: batch[1]})
            print("Epoch: {}, accuracy: {}, loss: {}".format(i, acc, loss))
            
     # evaluate our testing accuracy       
    acc = accuracy.eval(feed_dict = {x: X_test, y_: y_test})
    print("testing accuracy: {}".format(acc))
    data_test = np.apply_along_axis(lambda x: (x - m_v) / v_v, 1, data_test)
    #Run the DNN!
    results = sess.run(y, feed_dict = {x: data_test})
    #Conver the results into a single column vector with 0s and 1s
    predictions = np.apply_along_axis(lambda x: 0 if x[0] > x[1] else 1, 1, results)
    output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
    output.to_csv('titanic-predictions-DNN.csv', index = False)
    output.head()