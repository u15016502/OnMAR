import gc
from copy import deepcopy
from algorithm.parameters import params
from fitness.supervised_learning.supervised_learning import supervised_learning
from utilities.fitness.error_metric import ari
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utilities.fitness.math_functions import *
from fitness.supervised_learning.art_utils import OnlineFuzzyART
import sys 
from stats.stats import get_stats, stats
import ray
from os import path
import concurrent.futures
import os 
import signal
from contextlib import contextmanager
import tensorflow_hub as hub
from fitness.supervised_learning.exploratory_landscape_analysis import get_ela
import random
from representation.generate_grammar import create_grammar, GRAMMARS, FUNCTIONS
from representation import grammar
from representation import individual

def infer_cluster_labels(labels_, actual_labels):
    labels_ = labels_.astype(np.uint8)
    actual_labels = actual_labels.astype(np.uint8)

    inferred_labels = {}

    for i in range(len(list(np.unique(np.array(actual_labels))))):   
        labels = []
        index = np.where(labels_ == i)
        labels.append(actual_labels[index])

        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        try:
            if np.argmax(counts) in inferred_labels:
                inferred_labels[np.argmax(counts)].append(i)
            else:
                inferred_labels[np.argmax(counts)] = [i]
        except:
            counts = None
        
    return inferred_labels  

def infer_data_labels(X_labels, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
    returns: predicted labels for each array
    """
    
    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
    
    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
                
    return predicted_labels

@ray.remote(num_cpus=0.25)
def run_fuzzy_art(rho, alpha, beta, x_data, y_data, choice_fn, seed):
    # print("Evaluating individual: %s" % individual)
    # Create a FuzzyART network with the individual's category choice function

    fa = OnlineFuzzyART(rho, alpha, beta, x_data.shape[1], choice_fn=choice_fn)
    # Run the clustering on the dataset and find the clusters

    iterations, train_clusters, test_clusters = fa.run_batch(x_data, y_data, max_epochs=20, seed=seed)

    return iterations, train_clusters, test_clusters

def embed_sentences(sentences):
    # https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/
    module_url = "/Users/miagerber/Documents/MSc/Code/Clustering version/feature-extraction/universal-sentence-encoder_4"
    model = hub.load(module_url)
    query_vec = model(sentences)
    return query_vec

def get_features_and_cluster(ind, choice_funcs=[]):
    fitnesses = []
    all_choice_functions = []

    choice_fns = ind.phenotype.split('$')
    
    # for phenotype in ind.phenotype.split('$'):
    #     choice_fns = []


    #     if ind.num_funcs > 1:
    #         prev_func = phenotype
    #         choice_fns_ = deepcopy(ind.best_from_ela)

    #         for i_f, f in enumerate(ind.when_to_change):
    #             if i_f == 0 or f == 0:
    #                 choice_fns.append(prev_func)
    #             else:
    #                 prev_func = choice_fns_.pop()
    #                 choice_fns.append(prev_func)
    #     else:
    #         epochs = len(ind.when_to_change)
    #         choice_fns = [phenotype] * epochs

    fold = random.choice([0,1,2,3,4,5])#int(stats['gen']) % 9
    size = 150

    x_data = params['TRAINING_IN'][(size * fold): (size * (fold + 1))]
    x_class = params['TRAINING_EXP'][(size * fold): (size * (fold + 1))]
    y_data = params['TEST_IN'][(size * fold): (size * (fold + 1))]
    y_class = params['TEST_EXP'][(size * fold): (size * (fold + 1))]

    rho = float(params['EXTRA_PARAMETERS'][0])
    alpha = float(params['EXTRA_PARAMETERS'][1]) 
    beta = float(params['EXTRA_PARAMETERS'][2])

    try:

        fa = OnlineFuzzyART(rho, alpha, beta, x_data.shape[1], choice_fns=choice_fns)
        iterations, train_clusters, test_clusters, train_data, test_data = fa.run_batch(x_data, y_data, max_epochs=12)

        cluster_labels = infer_cluster_labels(test_clusters, y_class)
        predicted_labels = infer_data_labels(test_clusters, cluster_labels)


        accs = accuracy_score(y_class, predicted_labels)
        f1s = f1_score(y_class, predicted_labels, average='weighted')
        precs = precision_score(y_class, predicted_labels, average='weighted')
        recs = recall_score(y_class, predicted_labels, average='weighted')

        # print('accuracy\t' + str(accs))
        # print('f1\t' + str(f1s))
        # print('precision\t' + str(precs))
        # print('recall\t' + str(recs))
    except Exception as e:
        print(e)
        accs = -1

    #     fitnesses.append(accs)
    #     all_choice_functions.append(choice_fns)

    # ind.fitnesses = fitnesses
    # ind.fitness = max(fitnesses)

    # if ind.part_of_ela == False:
    #     hist, best = get_ela(choice_fns)
    #     ind.ela_history.append(hist)

    #     if len(ind.ela_history) > 1:
    #         should_change = abs(ind.ela_history[-2]['ela_conv.lin_dev_orig']) < abs(ind.ela_history[-1]['ela_conv.lin_dev_orig'])
    #     else:
    #         should_change = 0
            
    #     ind.when_to_change.append(int(should_change))
    #     ind.num_funcs += int(should_change)

    #     if should_change:
    #         ind.best_from_ela.append(best)

    #     print(str(all_choice_functions[fitnesses.index(max(fitnesses))]) + '\t\t\t' + str(ind.fitness))


    ind.fitness = accs
    return ind.fitness

class art(supervised_learning):
    """Fitness function for classification. We just slightly specialise the
    function for supervised_learning."""

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        # Set error metric if it's not set already.
        if params['ERROR_METRIC'] is None:
            params['ERROR_METRIC'] = f1_score

        self.maximise = True#params['ERROR_METRIC'].maximise

    def evaluate(self, ind, choice_funcs):
        """
        Note that math functions used in the solutions are imported from either
        utilities.fitness.math_functions or called from numpy.

        :param ind: An individual to be evaluated.
        :param kwargs: An optional parameter for problems with training/test
        data. Specifies the distribution (i.e. training or test) upon which
        evaluation is to be performed.
        :return: The fitness of the evaluated individual.
        """

        return get_features_and_cluster(ind, choice_funcs)