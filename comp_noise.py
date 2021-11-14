"""A multi-layer perceptron for classification of MNIST handwritten digits."""
from __future__ import absolute_import, division
from __future__ import print_function

from jax.api import jit, grad
from jax.config import config
from jax import hessian as hess
import jax.numpy as np
import jax.random as random
from jax.flatten_util import ravel_pytree
import numpy as onp
import sys

JAX_ENABLE_X64=True
np.set_printoptions(threshold=sys.maxsize)

def gen_data_y(params, inputs):
    print("Generating Y data values: ")
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = sigmoid(outputs)
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
    for W, b in params[0:2]:
        outputs = np.dot(inputs, W) + b
        inputs = sigmoid(outputs)   # This doesn't affect the final output of the last layer
    for W, b in params[2:]:
        outputs = np.dot(inputs, W) + b
        inputs = outputs
    return outputs 

def mean_square_error(params, inputs, targets):
    net_out = neural_net_predict(params, inputs)
    return (1/inputs.shape[0])*np.sum(np.power((net_out - targets),2))

def mean_square_error_model(params, inputs, targets):
    net_out = neural_net_predict_model(params, inputs)
    return (1/inputs.shape[0])*np.sum(np.power((net_out - targets),2))

def mean_square_error_model_flat(flat_params, inputs, targets, unflattener):
    unflat_params = unflattener(flat_params)
    return mean_square_error_model(unflat_params, inputs, targets)

def MDL_track(model_params, train_data, train_labels, noise_covar, hess_file, batch):
    model_like = -(1/(2*noise_covar))*np.power(train_labels-neural_net_predict_model(model_params, train_data), 2)
    model_params_flat, unflattener = ravel_pytree(model_params)
    hessian = hess(mean_square_error_model_flat)(model_params_flat, batch[0], batch[1], unflattener)
    hess_eig_values, hess_eig_vecs = onp.linalg.eig(hessian)
    hess_file.write(str(hess_eig_values) + '\n')
    sum_eig_values = np.sum(hess_eig_values)
    abs_sum_eig_values = np.sum(np.abs(hess_eig_values))
    trunc_eig_values = np.array([])
    for i in range(len(hess_eig_values)):
        if hess_eig_values[i].real > 1.0:                                       # NOTE check that I can exclude imag part
            trunc_eig_values = np.append(trunc_eig_values, hess_eig_values[i].real)
    sum_eig_kept = np.sum(trunc_eig_values)
    abs_sum_eig_kept = np.sum(np.abs(trunc_eig_values))
    prop_kept = sum_eig_kept/sum_eig_values
    abs_prop_kept = abs_sum_eig_kept/abs_sum_eig_values
    data_prob_from_model = (1/np.sqrt(2*np.pi*noise_covar))*np.exp(model_like) #NOTE may have to use noise_covar for normalize variance
    log_data_prob = np.log(data_prob_from_model)
    shannon_entropy = np.sum(data_prob_from_model) * -np.sum(log_data_prob)  #NOTE this is not entirely correct, need to be datawise 1st
    #shannon_entropy = -np.sum(data_prob_from_model * (model_like - (np.pi*covar)))
    model_entropy = 1/np.sqrt(np.prod(trunc_eig_values))
    data_entropy = model_entropy + shannon_entropy
    entropy_sign = np.prod(np.where(hess_eig_values > 0, 1, -1))
    return np.array([shannon_entropy, model_entropy, data_entropy, prop_kept, abs_prop_kept, entropy_sign])

def track_errors(model_params, train_data, train_labels, clean_train_labels, noise_covar, test_data, test_labels):
    full_data = np.vstack([train_data, test_data])
    full_labels = np.vstack([clean_train_labels, test_labels])
    train_model_error = np.mean(np.power(train_labels-neural_net_predict_model(model_params, train_data), 2))
    test_model_error = np.mean(np.power(test_labels-neural_net_predict_model(model_params, test_data), 2))
    full_model_error = np.mean(np.power(full_labels-neural_net_predict_model(model_params, full_data), 2))
    return np.array([train_model_error, test_model_error, full_model_error])

def main():
    # Hyper Parameters
    true_param_scale = 1.5 #1.2 #1.0
    net_param_scale = 0.8 #0.7 #1.5 #0.8
    noise_param_scale = 0.0 #0.3 #0.2
    batch_size = 50
    num_epochs = 300
    step_size = 0.04 #0.07 #0.01
    key = random.PRNGKey(onp.random.randint(0,100000000))

    # Model parameters
    true_layers = onp.array([50, 8, 1])
    noise_layers = [50, 1]
    layer_sizes = onp.array([50, 20, 12, 7, 1])
    true_param_position = random.uniform(key, shape=(1,), dtype=onp.float64, minval=-0.5, maxval=0.5)
    true_model = init_random_params(key, true_param_position, true_param_scale, true_layers)
    noise_model = init_random_params(key, 0.0, noise_param_scale, noise_layers)

    print("Loading data...")
    train_data = random.uniform(key, shape=(50,50), dtype=onp.float64, minval=0.0, maxval=1.0) #(500,100)
    test_data = random.uniform(key, shape=(1000,50), dtype=onp.float64, minval=0.0, maxval=1.0) #(500, 100)
    noise_covar = random.uniform(key, shape=(layer_sizes[-1],layer_sizes[-1]), dtype=onp.float64, minval=0.32, maxval=0.32)#0.35 0.35 #0.6 0.6
    train_labels_clean = gen_data_y(true_model, train_data)
    train_noise = gen_data_noise(key, noise_model, train_data, noise_covar)
    train_labels = train_labels_clean + train_noise
    test_labels = gen_data_y(true_model, test_data)
    train_true_error = np.mean(np.power(train_labels-neural_net_predict(true_model, train_data), 2))
    print("Test labels mean: ", np.mean(np.abs(test_labels)))

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

    def data_stream_clean():
        rng = onp.random.RandomState()
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]                                                                    
                yield train_data[batch_idx], train_labels_clean[batch_idx]
    batches_clean = data_stream_clean()

    @jit
    def update(params, batch, step_size):
        grads = grad(mean_square_error_model)(params, batch[0], batch[1])                                                                
        return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)] 

    params = init_random_params(key, 0.0, net_param_scale, layer_sizes)

    hess_file = open('hess.txt', 'w') 
    passed_threshold = False

    print("     Epoch     |      Train loss    |     Test loss     |   Dist Train Loss  | Dist Test Loss")
    for i in range(1,num_epochs):
        for _ in range(num_batches):
            if not passed_threshold:
                params = update(params, next(batches_clean), step_size)
                mdl_point = MDL_track(params, train_data, train_labels, noise_covar, hess_file, next(batches_clean))
                err_point = track_errors(params, train_data, train_labels, train_labels_clean,noise_covar, test_data, test_labels)
            else:
                params = update(params, next(batches), step_size)
                mdl_point = MDL_track(params, train_data, train_labels, noise_covar, hess_file, next(batches))
                err_point = track_errors(params, train_data, train_labels, train_labels_clean,noise_covar, test_data, test_labels)
            print("{:15}|{:20}|{:20}|{:20}|{:20}".format(i, err_point[0], err_point[1], train_true_error, 0.0))
            
            if err_point[0] < 0.4 and passed_threshold == False:
                print("Passed error threshold")
                passed_threshold = True

    hess_file.close()

if __name__ == '__main__':
    main()
