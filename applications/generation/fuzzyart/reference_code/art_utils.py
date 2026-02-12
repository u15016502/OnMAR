import enum
import numpy as np
from numpy import linalg as la
import random
import sys
import concurrent.futures
import os 

def run_compiled(args):
    func = args['func']
    default_func = args['default']
    res_arr = []

    for jx in args['chunked_indexes']:
        res_arr.append((jx, 
        # default_func(args['x'], args['w'][jx, :], args['alpha'])
        
        eval(str(func), globals(), {
                            'x': args['x'],
                            'jx': jx,
                            'w': args['w'][jx, :],
                            'alpha':args['alpha'],
                            'l1norm':args['l1norm'],
                            'nc':args['nc'],
                            'epoch':args['epoch'],
                            'alpha':args['alpha'],
                            'beta':args['beta'],
                            'rho':args['rho'],
                            'fuzzy_and':args['fuzzy_and'],
                            'fuzzy_or':args['fuzzy_or'],
                            'median':args['median'],
                            'avg':args['avg'],
                            'l2norm':args['l2norm'],
                            'random_decimal':args['random_decimal'],
                            'add':args['add'],
                            'sub':args['sub'],
                            'mul':args['mul'],
                            'div':args['div'],
                            'min':args['min'],
                            'max':args['max'],
                            'exp':args['exp'],
                            # 'inner':args['inner'],
                            # 'outer':args['outer'],
                            'matmul':args['matmul'],
                            # 'dot':args['dot'],
                            'sin':args['sin'],
                            'cos':args['cos'],
                            'tan':args['tan'],
                            'log':args['log'],
                            'stdev':args['stdev'],
                            'var':args['var'],
                            'average':args['average'],
                            'median':args['median'],
                            'logical_and':args['logical_and'],
                            'logical_or':args['logical_or'],
                            'logical_not':args['logical_not'],
                            'logical_xor':args['logical_xor'],
                        })
                        ))

    return res_arr

def chomp(line):
    """
    remove newline characters from line

    :param line: the line to remove the newline characters from
    :type line: str
    :return: the line with any newline characters removed
    :rtype: str
    """
    return line.replace('\n', '')


def split(line, separator=' '):
    """
    Splits line by separator and returns the list of parts

    The line is passed through chomp first to get rid of newline characters

    :param line: the line to split
    :type line: str
    :param separator: the separator character(s), defaults to space
    :type separator: str
    :return: a list of strings that the line splits off into
    :rtype: list
    """
    return chomp(line).split(separator)


def cat_to_bin(values, choices):
    """
    convert a list of categorical field values to a binary occupancy list format

    :param values: the categorical field values
    :type values: list
    :param choices: the category choices to represent
    :type choices: list
    :return: an occupancy list representation of values
    :rtype: list
    """
    result = [0] * len(choices)
    for value in values:
        if value not in choices:
            return None
        result[choices.index(value)] = 1
    return result


def scale_range(x, x_range, y_range=(0.0, 1.0)):
    """
    scale the number x from the range specified by x_range to the range specified by y_range

    :param x: the number to scale
    :type x: float
    :param x_range: the number range that x belongs to
    :type x_range: tuple
    :param y_range: the number range to convert x to, defaults to (0.0, 1.0)
    :type y_range: tuple
    :return: the scaled value
    :rtype: float
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    return (y_max - y_min) * (x - x_min) / (x_max - x_min) + y_min

def divide_(x1, x2):
    try:
        return np.divide(x1, x2)
    except:
        return x1

def l2norm(x):
    return la.norm(x, ord=1)

def median(x, y):
	return np.median(np.array([x, y]), 0)

def avg(x,y):
	return np.average(np.array([x, y]), 0)

def max_norm(x):
    # noinspection PyTypeChecker
    return la.norm(x, ord=2)

def fuzzy_and(x, y):
    return np.min(np.array([x, y]), 0)

def fuzzy_or(vector1, vector2):
	return np.max(np.array([vector1, vector2]), 0)

def default_category_choice(pattern, category_w, alpha):
    return 'l1norm(fuzzy_and(x, w)) / (alpha + l1norm(w))'

class OnlineFuzzyART(object):
    def __init__(self, rho, alpha, beta, num_features, choice_fns=['l1norm(fuzzy_and(x, w)) / (alpha + l1norm(w))'], w=None):
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.num_features = num_features
        self.w = w if w is not None else np.ones((1, num_features * 2))
        self.num_clusters = self.w.shape[0] - 1
        self.clusters = np.zeros(0)
        self.choice_fns = choice_fns
        self.choice_fn = choice_fns[0]

    def run_batch(self, dataset, test_set, max_epochs=np.inf, seed=None,dataset_name=None,grammar_type=None,metric=None):
        # complement-code the data
        dataset = np.concatenate((dataset, 1 - dataset), axis=1)
        test_set = np.concatenate((test_set, 1 - test_set), axis=1)
        # compiled = compile(self.choice_fn, '<string>', 'eval')

        # initialize variables
        cluster_choices = np.zeros(dataset.shape[0])
        iterations = 0
        w_old = None

        if seed is not None:
            random.seed(seed)

            # repeat the learning until either convergence or max_epochs
        # while not np.array_equal(self.w, w_old) and iterations < max_epochs:
        while iterations < max_epochs:#not np.array_equal(self.w, w_old) and iterations < max_epochs:
            # self.choice_fn = self.choice_fns[iterations]

            w_old = self.w
            cluster_choices = np.zeros(dataset.shape[0])

            indices = list(range(dataset.shape[0]))
            random.shuffle(indices)

            # present the input patters to the Fuzzy ART module
            for ix in indices:
                cluster_choices[ix] = self.train_pattern((dataset[ix, :], iterations))

            # with concurrent.futures.ProcessPoolExecutor() as executor:
            #     futures = [executor.submit(self.train_pattern, ( dataset[ix, :], iterations )) for ix in indices]

            # results = [f.result() for f in futures]

            # for idx, ix in enumerate(indices):
            #     cluster_choices[ix] = results[idx]

            iterations += 1
            # print(iterations)

            # print(str(iterations) + '\t' + dataset_name + '\t' + grammar_type + '\t' + metric + '\t' + str(list(cluster_choices)).replace('[','').replace(']','').replace(',','\t'))

        # return results

        train_cluster_choices = cluster_choices

        cluster_choices = np.zeros(test_set.shape[0])
        indices = list(range(test_set.shape[0]))
        random.shuffle(indices)

        # present the input patters to the Fuzzy ART module
        for ix in indices:
            cluster_choices[ix] = self.train_pattern((test_set[ix, :], iterations))

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #    futures = [executor.submit(self.train_pattern, ( test_set[ix, :], iterations )) for ix in indices]

        # results = [f.result() for f in futures]

        # for idx, ix in enumerate(indices):
        #    cluster_choices[ix] = results[idx]

        test_cluster_choices = cluster_choices

        return iterations, train_cluster_choices, test_cluster_choices, dataset, test_set

    def run_online(self, data_reader, data_ranges, max_epochs=np.inf, seed=None):
        # initialize variables
        cluster_choices = np.zeros(len(data_reader))
        iterations = 0
        w_old = None

        if seed is not None:
            random.seed(seed)

        def normalize(p):
            for i in range(len(p)):
                p[i] = scale_range(p[i], data_ranges[i])

        # repeat the learning until either convergence or max_epochs
        while iterations < max_epochs:#not np.array_equal(self.w, w_old) and iterations < max_epochs:
            w_old = self.w
            indices = list(range(len(data_reader)))
            random.shuffle(indices)
            for ix in indices:
                pattern = np.array(data_reader[ix], dtype=float)
                normalize(pattern)
                pattern = np.concatenate((pattern, 1.0 - pattern))
                choice = self.train_pattern(pattern)
                cluster_choices[ix] = choice
            iterations += 1

        # return results
        return iterations, np.array(cluster_choices)

    def train_pattern(self, tup): #pattern, iteration):
        # evaluate the pattern to get the winning category
        pattern = tup[0]
        iteration = tup[1]
        winner = self.eval_pattern(pattern, iteration)

        # update the weight of the winning neuron
        try:
            self.w[winner, :] = self.beta * fuzzy_and(pattern, self.w[winner, :]) + (1 - self.beta) * self.w[winner, :]
        except:
            try:
                self.w[winner, :] = self.beta * np.log(fuzzy_and(pattern, self.w[winner, :])) + (1 - self.beta) * np.log(self.w[winner, :])
            except:
                try:
                    self.w[winner, :] = self.beta * fuzzy_and(np.log(pattern), np.log(self.w[winner, :])) + (1 - self.beta) * np.log(self.w[winner, :])
                except:
                    sys.exc_info()[0]
                    sys.exc_info()[1]

        # check if the uncommitted node was the winner
        if (winner + 1) > self.num_clusters:
            self.num_clusters += 1
            self.w = np.concatenate((self.w, np.ones((1, self.w.shape[1]))))

        return winner


    def eval_pattern(self, pattern, iteration):
        # initialize variables
        matches = np.zeros(self.w.shape[0])
        # calculate the category match values
        all = range(self.w.shape[0])
        
        results = run_compiled({
                        'default': default_category_choice,
                        'chunked_indexes': all,
                        'func': self.choice_fn,
                        'x': pattern,
                        'w': self.w,
                        'alpha': self.alpha,
                        'l1norm': max_norm,
                        'nc': self.num_clusters,
                        'epoch': iteration,
                        'alpha': self.alpha,
                        'beta': self.beta,
                        'rho': self.rho,
                        'fuzzy_and': fuzzy_and,
                        'fuzzy_or': fuzzy_or,
                        'median': median,
                        'avg': avg,
                        'l2norm': l2norm,
                        'random_decimal': random.random,
                        'add': np.add,
                        'sub': np.subtract,
                        'mul': np.multiply,
                        'div': divide_,
                        'min': np.min,
                        'max': np.max,
                        'exp': np.exp,
                        # 'inner': np.inner,
                        # 'outer': np.outer,
                        'matmul': np.matmul,
                        # 'dot': np.dot,
                        'sin': np.sin,
                        'cos': np.cos,
                        'tan': np.tan,
                        'log': np.log,
                        'stdev': np.std,
                        'var': np.var,
                        'average': np.average,
                        'median': np.median,
                        'logical_and': np.logical_and,
                        'logical_or': np.logical_or,
                        'logical_not': np.logical_not,
                        'logical_xor': np.logical_xor,
                    })

        for (jx, val) in results:
            if 'list' in str(type(val)) or 'ndarray' in str(type(val)):
                matches[jx] = np.average(np.array(val))
            else:
                matches[jx] = val

        vigilance_test = self.rho * max_norm(pattern)
        match_attempts = 0
        while match_attempts < len(matches):
            # winner-take-all selection
            winner = np.argmax(matches)
            # vigilance test
            if max_norm(fuzzy_and(pattern, self.w[winner, :])) >= vigilance_test:
                # the winning category passed the vigilance test
                return winner
            else:
                # shut off this category from further testing
                matches[winner] = 0
                match_attempts += 1
        return len(matches) - 1


