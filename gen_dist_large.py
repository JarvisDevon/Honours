from __future__ import absolute_import, division
from __future__ import print_function
import matplotlib.pyplot as plt

from jax.api import jit, grad
from jax.config import config
import jax.numpy as np
import jax.random as random
import numpy as onp
import sys

JAX_ENABLE_X64=True
np.set_printoptions(threshold=sys.maxsize)

def gen_data_y(params, inputs):
    print("Generating Y data values: ")
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = sigmoid(outputs)
    print("Max abs pure y generated: ", np.max(np.abs(outputs)))
    return outputs

def gen_data_noise(key, params, inputs, noise_covariance):
    print("Generating Y data noise: ")
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = sigmoid(outputs)
    total_noise = outputs + random.multivariate_normal(key, np.zeros(outputs.shape[1]), noise_covariance, (outputs.shape[0],))
    print("Sum noise model noise", np.sum(np.abs(outputs)))
    print("Sum noise: ", np.sum(total_noise))
    print("Abs Sum noise: ", np.sum(np.abs(total_noise)))
    print("Noise shape:", total_noise.shape)
    print("Max abs noise: ", np.max(np.abs(total_noise)))
    return total_noise

def init_random_params(key, position, scale, layer_sizes):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(random.uniform(key, shape=(m,n), dtype=onp.float64, minval=-scale, maxval=scale)+position, #weight matrix
            random.uniform(key, shape=(n,), dtype=onp.float64, minval=-scale, maxval=scale)+position) #bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)),  np.exp(x)/(1+np.exp(x)))

def softmax(x):
    return np.exp(x-np.max(x, axis=1).reshape(x.shape[0],1))/np.sum(np.exp(x-np.max(x, axis=1).reshape(x.shape[0],1)), axis=1).reshape(x.shape[0],1)

def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix."""
    #print("Their params: ###########################################################################################")
    #print(params)
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = sigmoid(outputs)   # This doesn't affect the final output of the last layer
    return outputs 

def neural_net_predict_model(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix."""
    for W, b in params[0:len(params)-2]:
        outputs = np.dot(inputs, W) + b
        inputs = sigmoid(outputs)   # This doesn't affect the final output of the last layer
    for W, b in params[len(params)-2:]:
        outputs = np.dot(inputs, W) + b
        inputs = outputs
    return outputs 

def mean_square_error(params, inputs, targets):
    net_out = neural_net_predict(params, inputs)
    return (1/inputs.shape[0])*np.sum(np.power((net_out - targets),2))

def mean_square_error_model(params, inputs, targets):
    net_out = neural_net_predict_model(params, inputs)
    return (1/inputs.shape[0])*np.sum(np.power((net_out - targets),2))

def relative_error(params, inputs, targets):
    net_out = neural_net_predict(params, inputs)
    return np.mean(np.abs((net_out - targets)/targets))

def accuracy(params, inputs, targets):
    target_class    = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)

def jeffreys_dist(model_params, true_params, train_data, train_labels, noise_covar, test_data, test_labels, itr):
    model_like = np.sum(-(1/(2*noise_covar))*np.power(train_labels-neural_net_predict_model(model_params, train_data), 2))
    true_like = np.sum(-(1/(2*noise_covar))*np.power(train_labels-neural_net_predict(true_params, train_data), 2))
    return np.array([model_like, true_like])

def track_errors(model_params, train_data, train_labels, noise_covar, test_data, test_labels, itr):
    train_model_error = np.mean(np.power(train_labels-neural_net_predict_model(model_params, train_data), 2))
    test_model_error = np.mean(np.power(test_labels-neural_net_predict_model(model_params, test_data), 2))
    rel_model_error = np.mean(np.abs(neural_net_predict_model(model_params, test_data) - test_labels/test_labels))
    return np.array([train_model_error, test_model_error, rel_model_error])

def main():
    # Hyper Parameters
    true_param_scale = 1.5
    net_param_scale = 1.0
    noise_param_scale = 0.0
    batch_size = 50
    num_epochs = 300
    #step_size = 0.0008 #0.0003 #*0.0005

    track_file = open('trainings.txt', 'w')

    for k in range(1000):
        log_file = open('most_recent_log_' + str(k) + '.txt', 'w')
        log_file.write("Model Likelihoods | True Likelihoods | Train Errors | Test Errors\n")
        print("Training ", k)
        key = random.PRNGKey(onp.random.randint(0,100000000))

        # Model parameters
        true_net_size = onp.random.randint(5,15)
        print("True net size: ", true_net_size)
        model_net_size = onp.random.randint(true_net_size+5,25) # Make sure model is deeper than true net
        print("Model net size: ", model_net_size)
        noise_layers = [100, 1]
        true_layers = onp.append(onp.random.randint(5,100, true_net_size), 1)
        true_layers.sort()
        true_layers = true_layers[::-1]
        true_layers[0] = 100
        print("True model layer sizes: ", true_layers)
        layer_sizes = onp.append(onp.random.randint(true_layers[-2],100, model_net_size), 1) #Make sure model doesn't have smaller layers
        layer_sizes.sort()
        layer_sizes = layer_sizes[::-1]
        layer_sizes[0] = 100
        print("Train model layer sizes: ", layer_sizes)
        true_param_position = random.uniform(key, shape=(1,), dtype=onp.float64, minval=-0.5, maxval=0.5) #*-0.5 0.5 #-1.0 1.0
        print("True Params position: ", true_param_position)
        true_model = init_random_params(key, true_param_position, true_param_scale, true_layers)
        noise_model = init_random_params(key, 0.0, noise_param_scale, noise_layers)

        print("Loading data...")
        train_data = random.uniform(key, shape=(50,100), dtype=onp.float64, minval=0.0, maxval=1.0) #(500,100) maxval=1.0
        test_data = random.uniform(key, shape=(1000,100), dtype=onp.float64, minval=0.0, maxval=1.0) #(500, 100) maxval=1.0
        noise_covar = random.uniform(key, shape=(layer_sizes[-1],layer_sizes[-1]), dtype=onp.float64, minval=0.6, maxval=0.6)#0.2 0.2#0.4 0.4*
        noise_for_train_labels = gen_data_noise(key, noise_model, train_data, noise_covar)
        train_labels_clean = gen_data_y(true_model, train_data)
        train_labels = train_labels_clean + noise_for_train_labels
        test_labels = gen_data_y(true_model, test_data)
        print("Test labels mean: ", np.mean(np.abs(test_labels)))
        print("test labels abs max: ", np.max(np.abs(test_labels)))
        print("Test labels variance: ", np.var(test_labels))

        num_train = train_data.shape[0]
        num_complete_batches, leftover = divmod(num_train, batch_size)
        num_batches = num_complete_batches + bool(leftover)
        
        def data_stream():
            rng = onp.random.RandomState()
            while True:
                perm = rng.permutation(num_train)
                for i in range(num_batches):
                    batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                    yield train_data[batch_idx], train_labels[batch_idx]
        batches = data_stream()

        @jit
        def update(params, batch, step_size):
            grads = grad(mean_square_error_model)(params, batch[0], batch[1])
            return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]

        likes_cross_points = np.array([])
        error_turn_points = np.array([])
        jeff_errors_list = np.array([])
        diff_errors_list = np.array([])
        min_test_errors = np.array([])
        likes_diffs = np.array([])
        rel_jeff_errors_list = np.array([])
        rel_min_test_errors = np.array([])
        rel_diff_errors_list = np.array([])

        for step_size in [0.0008]: #onp.arange(0.0001, 0.0011, 0.0002):
            print("Training with step_size: ", step_size)
            params = init_random_params(key, 0.0, net_param_scale, layer_sizes)

            model_likelihoods = np.array([])
            true_likelihoods = np.array([])
            train_errors = np.array([])
            test_errors = np.array([])
            relative_errors = np.array([])

            print("     Epoch     |      Train loss    |     Test loss     |   Dist Train Loss  | Dist Test Loss")
            for i in range(1,num_epochs):
                for _ in range(num_batches):
                    params = update(params, next(batches), step_size)
                    jd_point = jeffreys_dist(params, true_model, train_data, train_labels, noise_covar, test_data, test_labels, i)
                    err_point = track_errors(params, train_data, train_labels, noise_covar, test_data, test_labels, i)
                    model_likelihoods = np.append(model_likelihoods, jd_point[0])
                    true_likelihoods = np.append(true_likelihoods, jd_point[1])
                    train_errors = np.append(train_errors, err_point[0])
                    test_errors = np.append(test_errors, err_point[1])
                    relative_errors = np.append(relative_errors, err_point[2])

                train_loss = mean_square_error_model(params, train_data, train_labels)
                test_loss  = mean_square_error_model(params, test_data, test_labels)
                true_train_loss = mean_square_error(true_model, train_data, train_labels)
                true_test_loss  = mean_square_error(true_model, test_data, test_labels)
                rel_train_loss = relative_error(params, train_data, train_labels)
                rel_test_loss  = relative_error(params, test_data, test_labels)
                rel_true_train_loss = relative_error(true_model, train_data, train_labels)
                rel_true_test_loss  = relative_error(true_model, test_data, test_labels)
                print("{:15}|{:20}|{:20}|{:20}|{:20}".format(i, train_loss, test_loss, true_train_loss, true_test_loss))
                log_file.write(str(i)+","+str(train_loss)+","+str(test_loss)+","+str(true_train_loss)+","+str(true_test_loss)+
                        ","+str(rel_train_loss)+","+str(rel_test_loss)+","+str(rel_true_train_loss)+","+str(rel_true_test_loss)+"\n")

            log_file.flush()
            like_diffs = np.abs(true_likelihoods - model_likelihoods)
            likes_cross = np.argmin(like_diffs)
            error_turn = np.argmin(test_errors)
            jeff_error = test_errors[likes_cross]
            min_test_error = test_errors[error_turn]
            diff_errors = jeff_error - test_errors[error_turn]
            rel_jeff_error = relative_errors[likes_cross]
            min_rel_error = relative_errors[error_turn]
            diff_rel_error = rel_jeff_error - relative_errors[error_turn]

            likes_cross_points = np.append(likes_cross_points, likes_cross)
            error_turn_points = np.append(error_turn_points, error_turn)
            jeff_errors_list = np.append(jeff_errors_list, jeff_error)
            diff_errors_list = np.append(diff_errors_list, diff_errors)
            min_test_errors = np.append(min_test_errors, test_errors[error_turn])
            likes_diffs = np.append(likes_diffs, like_diffs[likes_cross])
            rel_jeff_errors_list = np.append(rel_jeff_errors_list, rel_jeff_error)
            rel_min_test_errors = np.append(rel_min_test_errors, min_rel_error)
            rel_diff_errors_list = np.append(rel_diff_errors_list, diff_rel_error)
            
            print("Found likelihood cross point: ", likes_cross)
            print("Found error min point: ", error_turn)
            print("Difference in Jeffreys error and min error: ", diff_errors)
            print("Minimum test error: ", test_errors[error_turn])
            print("Difference in likelihoods at cross point: ", like_diffs[likes_cross])
            print("Jeffreys Relative Error: ", rel_jeff_error)
            print("Minimum MSE point relative error: ", min_rel_error)
            print("Difference in Jeffreys relative error and min MSE error point relative error: ", diff_rel_error)

        best_test_error_training = np.argmin(min_test_errors)
        print("List of minimum test errors: ", min_test_errors)
        #print("Best training learning rate: ", onp.arange(0.0001, .0011, 0.0001)[best_test_error_training])
        best_likes_cross = likes_cross_points[best_test_error_training]
        best_error_turn = error_turn_points[best_test_error_training]
        best_diff_errors = diff_errors_list[best_test_error_training]
        best_min_test_error = min_test_errors[best_test_error_training]
        best_likes_diff = likes_diffs[best_test_error_training]
        best_rel_jeff_error = rel_jeff_errors_list[best_test_error_training]
        best_min_rel_error = rel_min_test_errors[best_test_error_training]
        best_rel_diff_error = rel_diff_errors_list[best_test_error_training]
        training_data_variance = np.var(train_labels_clean)
        noise_for_train_data_variance = np.var(noise_for_train_labels)
        test_data_variance = np.var(test_labels)

        print("Found best likelihood cross point: ", best_likes_cross)
        print("Found best  error min point: ", best_error_turn)
        print("Best difference in Jeffreys error and min error: ", best_diff_errors)
        print("Best minimum test error: ", best_min_test_error)
        print("Best difference in likelihoods at cross point: ", best_likes_diff)
        track_file.write(str(k) + "," + str(best_likes_cross) + "," + str(best_error_turn) + "," + str(best_diff_errors) + "," 
                + str(best_min_test_error) + "," + str(best_likes_diff) + "," + str(true_net_size) + "," + str(model_net_size) +
                "," + str(best_rel_jeff_error) + "," + str(best_min_rel_error) + "," + str(best_rel_diff_error) + "," + 
                str(training_data_variance) + "," + str(test_data_variance) + "," + str(noise_for_train_data_variance) + "\n")
        track_file.flush()
        log_file.close()

if __name__ == '__main__':
    main()
