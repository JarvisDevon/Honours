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

def relative_error(params, inputs, targets):
    net_out = neural_net_predict(params, inputs)
    return (net_out - targets)/targets

def accuracy(params, inputs, targets):
    target_class    = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)

def jeffreys_dist(model_params, true_params, train_data, train_labels, noise_covar, test_data, test_labels, batch):
    full_data = np.vstack([train_data, test_data])
    full_labels = np.vstack([train_labels, test_labels])
    model_like = np.sum(-(1/(2*noise_covar))*np.power(train_labels-neural_net_predict_model(model_params, train_data), 2))
    model_params_flat, unflattener = ravel_pytree(model_params)
    hessian = hess(mean_square_error_model_flat)(model_params_flat, batch[0], batch[1], unflattener)
    hess_eig_values, hess_eig_vecs = onp.linalg.eig(hessian)
    trunc_eig_values = np.array([])
    for i in range(len(hess_eig_values)):
        if hess_eig_values[i].imag >= 0.05:
            print("ALERT: Found large imag component of eigenvalue=", hess_eig_values[i].image)
        #if hess_eig_values[i].real != 0.0 or hess_eig_values[i].image != 0.0:
        if hess_eig_values[i].real > 1.0:                                       # NOTE check that I can exclude imag part
            trunc_eig_values = np.append(trunc_eig_values, hess_eig_values[i].real)
    log_eig_values = np.log(trunc_eig_values)
    fisher_exponent = -0.5*np.sum(log_eig_values)
    true_like = np.sum(-(1/(2*noise_covar))*np.power(train_labels-neural_net_predict(true_params, train_data), 2))
    true_prob = (1/np.sqrt(2*np.pi*noise_covar))*np.exp(true_like)        # NOTE may have to switch to noise_covar for normalize variance
    model_prob = (1/np.sqrt(noise_covar))*np.exp(model_like-fisher_exponent) # The Laplace approx adds sqrt(2pi) to the normalizing
                                                                               # numerator. So I cancelled with sqrt(2pi) in denom
                                                                               # NOTE This should be verified
    jeff_dist = np.sum((true_prob - model_prob)*(true_like + fisher_exponent - model_like -np.log(np.sqrt(2*np.pi))))
    jeff_dist_excess = np.sum(true_like + fisher_exponent - model_like)
    
    # Calculating the point KL divergences for the parameter distribution (need entropy and model prior parts)
    KL_dist_true_to_model = true_prob*(np.log(true_prob*(1/model_prob)))
    KL_dist_model_to_true = model_prob*(np.log(model_prob*(1/true_prob)))

    # Calculating the KL divergences for the data distribution (only likelihoods needed in exp)
    model_indiv_likes = -(1/(2*noise_covar))*np.power(full_labels-neural_net_predict_model(model_params, full_data), 2)
    true_indiv_likes = -(1/(2*noise_covar))*np.power(full_labels-neural_net_predict(true_params, full_data), 2)
    model_indiv_probs = (1/np.sqrt(2*np.pi*noise_covar))*np.exp(model_indiv_likes)
    true_indiv_probs = (1/np.sqrt(2*np.pi*noise_covar))*np.exp(true_indiv_likes)
    KL_div_true_to_model = -np.sum(true_indiv_probs*np.log(true_indiv_probs*(1/model_indiv_probs)))
    KL_div_model_to_true = -np.sum(model_indiv_probs*np.log(model_indiv_probs*(1/true_indiv_probs)))
    square_train_disagreement = np.sum(np.power(neural_net_predict_model(model_params, train_data) - \
                                                                neural_net_predict(true_params, train_data), 2))
    square_test_disagreement = np.sum(np.power(neural_net_predict_model(model_params, test_data) - \
                                                                neural_net_predict(true_params, test_data), 2))
    return np.array([model_like, model_prob[0][0], fisher_exponent, true_like, true_prob[0][0], jeff_dist, \
                            square_train_disagreement, square_test_disagreement, jeff_dist_excess, \
                            KL_dist_true_to_model[0][0], KL_dist_model_to_true[0][0], KL_div_true_to_model, KL_div_model_to_true])

def MDL_track(model_params, train_data, train_labels, noise_covar, batch):
    model_like = -(1/(2*noise_covar))*np.power(train_labels-neural_net_predict_model(model_params, train_data), 2)
    model_params_flat, unflattener = ravel_pytree(model_params)
    hessian = hess(mean_square_error_model_flat)(model_params_flat, batch[0], batch[1], unflattener)
    hess_eig_values, hess_eig_vecs = onp.linalg.eig(hessian)
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
    true_param_scale = 1.5 #1.0
    net_param_scale = 0.8 #1.0 #1.5 #0.8
    noise_param_scale = 0.0 #0.3 #0.2
    batch_size = 50
    num_epochs = 101
    step_size = 0.07 #0.04 #0.07 #0.01
    key = random.PRNGKey(onp.random.randint(0,100000000))

    # Model parameters
    true_layers = onp.array([50, 8, 1]) #[25, 5, 1]
    noise_layers = [50, 1] #[25, 1]
    layer_sizes = onp.array([50, 20, 12, 7, 1]) #[25, 20, 12, 7, 1]
    true_param_position = random.uniform(key, shape=(1,), dtype=onp.float64, minval=-0.5, maxval=0.5)
    true_model = init_random_params(key, true_param_position, true_param_scale, true_layers)
    noise_model = init_random_params(key, 0.0, noise_param_scale, noise_layers)

    print("Loading data...")
    train_data = random.uniform(key, shape=(50,50), dtype=onp.float64, minval=0.0, maxval=1.0) #(500,25)
    test_data = random.uniform(key, shape=(1000,50), dtype=onp.float64, minval=0.0, maxval=1.0) #(1000, 25)
    noise_covar = random.uniform(key, shape=(layer_sizes[-1],layer_sizes[-1]), dtype=onp.float64, minval=0.4, maxval=0.4)#0.2 0.2#0.6 0.6
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

    @jit
    def update(params, batch, step_size):
        grads = grad(mean_square_error_model)(params, batch[0], batch[1])                                                                
        return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)] 

    params = init_random_params(key, 0.0, net_param_scale, layer_sizes)

    log_file = open('log.txt', 'w')
    log_file.write("jd_point = model_likelihoods, model_probabilities, fisher_info_exps, true_likelihoods, true_probabilities, jeffreys_distances, sqr_train_disagrees, sqr_test_disagrees, jeffreys_excesses, KL_distances_true_to_model, KL_distances_model_to_true, KL_divergences_true_to_model, KL_divergences_model_to_true \n")
    log_file.write("mdl_point = shannon_entropies, model_entropies, data_entropies, props_kept, abs_props_kept\n")
    log_file.write("err_point = train_errors, test_errors\n")
    log_file.write("jd_point  |   mdl_point  |  err_point \n")

    model_likelihoods = np.array([])
    model_probabilities = np.array([])
    fisher_info_exps = np.array([])
    true_likelihoods = np.array([])
    true_probabilities = np.array([])
    jeffreys_distances = np.array([])
    jeffreys_excesses = np.array([])
    sqr_train_disagrees = np.array([])
    sqr_test_disagrees = np.array([])
    KL_distances_true_to_model = np.array([])
    KL_distances_model_to_true = np.array([])
    KL_divergences_true_to_model = np.array([])
    KL_divergences_model_to_true = np.array([])
    shannon_entropies = np.array([])
    model_entropies = np.array([])
    data_entropies = np.array([])
    props_kept = np.array([])
    abs_props_kept = np.array([])
    entropy_signs = np.array([])
    full_hess_eigs = np.zeros((1,871))
    train_errors = np.array([])
    test_errors = np.array([])
    full_errors = np.array([])

    print("     Epoch     |      Train loss    |     Test loss     |   Dist Train Loss  | Dist Test Loss")
    for i in range(1,num_epochs):
        for _ in range(num_batches):         
            params = update(params, next(batches), step_size)
            jd_point = jeffreys_dist(params,true_model,train_data,train_labels,noise_covar,test_data,test_labels,next(batches))
            mdl_point = MDL_track(params, train_data, train_labels, noise_covar, next(batches))
            err_point = track_errors(params, train_data, train_labels, train_labels_clean,noise_covar, test_data, test_labels)
            print("{:15}|{:20}|{:20}|{:20}|{:20}".format(i, err_point[0], err_point[1], train_true_error, 0.0))
            
            model_likelihoods = np.append(model_likelihoods, jd_point[0])
            model_probabilities = np.append(model_probabilities, jd_point[1])
            fisher_info_exps = np.append(fisher_info_exps, jd_point[2])
            true_likelihoods = np.append(true_likelihoods, jd_point[3])
            true_probabilities = np.append(true_probabilities, jd_point[4])
            jeffreys_distances = np.append(jeffreys_distances, jd_point[5])
            sqr_train_disagrees = np.append(sqr_train_disagrees, jd_point[6])
            sqr_test_disagrees = np.append(sqr_test_disagrees, jd_point[7])
            jeffreys_excesses = np.append(jeffreys_excesses, jd_point[8])
            KL_distances_true_to_model = np.append(KL_distances_true_to_model, jd_point[9])
            KL_distances_model_to_true = np.append(KL_distances_model_to_true, jd_point[10])
            KL_divergences_true_to_model = np.append(KL_divergences_true_to_model, jd_point[11])
            KL_divergences_model_to_true = np.append(KL_divergences_model_to_true, jd_point[12])
            shannon_entropies = np.append(shannon_entropies, mdl_point[0])
            model_entropies = np.append(model_entropies, mdl_point[1])
            data_entropies = np.append(data_entropies, mdl_point[2])
            props_kept = np.append(props_kept, mdl_point[3])
            abs_props_kept = np.append(abs_props_kept, mdl_point[4])
            entropy_signs = np.append(entropy_signs, mdl_point[5])
            train_errors = np.append(train_errors, err_point[0])
            test_errors = np.append(test_errors, err_point[1])
            full_errors = np.append(full_errors, err_point[2])
            
            log_file.write(str(jd_point[0]) +"," + str(jd_point[1]) +"," + str(jd_point[2]) + "," + str(jd_point[3]) + "," + str(jd_point[4]) \
                    + "," + str(jd_point[5]) + "," + str(jd_point[6]) +"," + str(jd_point[7]) +"," + str(jd_point[8]) +"," + str(jd_point[9]) \
                    + "," + str(jd_point[10]) + "," + str(jd_point[11]) + "," + str(jd_point[12]) + "," + str(mdl_point[0]) \
                    + "," + str(mdl_point[1]) + "," + str(mdl_point[2]) + "," + str(mdl_point[3]) + "," + str(mdl_point[4]) \
                    + "," + str(mdl_point[5]) + "," + str(err_point[0]) + "," + str(err_point[1]) + "," + str(err_point[2]) \
                    + str(err_point[3]) + "\n")
 
    log_file.close()

if __name__ == '__main__':
    main()
