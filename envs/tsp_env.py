# [Yunqiu Xu] [20200317] a TSP environment class

import numpy as np
import pickle
import os
from matplotlib import pyplot as plt

class TSPProblem(object):
    def __init__(self, graph_size=20, seed=1234):
        self.graph_size = graph_size
        self.seed = seed


    def generate_batch_data(self, batch_size=1):
        """
        Generate a batch of tsp data: numpy.float64, (batch_size, graph_size, 2)
        """
        return {'loc': np.random.uniform(size=(batch_size, self.graph_size, 2))}


    def generate_test_dataset(self, dataset_size=1000, foldername="test_dataset/"):
        filename = "{}tsp{}_{}_seed{}.pkl".format(foldername, self.graph_size, dataset_size, self.seed)
        print("Generate test dataset at: {}".format(filename))
        np.random.seed(self.seed)
        dataset = []
        for i in range(dataset_size):
            curr_data = self.generate_batch_data(batch_size=1)
            dataset.append(curr_data)
        with open(filename, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        return filename


    def load_test_dataset(self, filename):
        """
        load existing dataset: numpy.float64, (n_samples, graph_size, 2) in uniform distrabution
        """
        assert os.path.splitext(filename)[1] == '.pkl', "Wrong path:{}".format(filename)
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
        print("Dataset loaded, it's a length-{}-list. During testing the batch_size is fixed as 1!".format(len(dataset)))
        print("Sample data:")
        sample_data = dataset[0]
        print("loc: {}, {}".format(sample_data['loc'].shape, sample_data['loc'].dtype))
        return dataset


    def compute_cost(self, inputs, selected_idxs):
        """
        :param inputs: a batch of tsp data, (batch_size, graph_size, 2)
        :param selected_idxs: a batch of selected idxs, (batch_size, graph_size), np.int
        :output: (batch_size,)
        """
        loc = inputs['loc']
        # get indexed data
        batch_size = loc.shape[0]
        indexed_data = loc[np.arange(batch_size)[:,None], selected_idxs]
        # compute cost
        cost_part1 = np.linalg.norm(indexed_data[:,1:] - indexed_data[:,:-1], axis=2).sum(1) 
        cost_part2 = np.linalg.norm(indexed_data[:,0] - indexed_data[:,-1], axis=1) # the last node and first node
        return cost_part1 + cost_part2


    def plot(self, inputs, selected_idxs, savepath=None):
        """
        https://github.com/wouterkool/attention-learn-to-route/blob/master/simple_tsp.ipynb
        savepath = "xyq_vis/tsp20_fig.png"
        """
        loc = inputs['loc']
        assert(loc.shape[0] == 1), "Only one problem"
        assert(selected_idxs.shape[0] == 1), "Only one problem"
        # get cost
        cost = self.compute_cost(inputs, selected_idxs)[0]
        # get indexed nodes
        indexed_data = loc[np.arange(1)[:,None], selected_idxs] # [1, graph_size, 2]
        xs = indexed_data[0,:,0]
        ys = indexed_data[0,:,1]
        dx = np.roll(xs, -1) - xs
        dy = np.roll(ys, -1) - ys
        # visualize
        plt.figure(figsize=(10,10))
        plt.xlim(0,1)
        plt.xticks([0.,0.2,0.4,0.6,0.8,1.])
        plt.ylim(0,1)
        plt.yticks([0.,0.2,0.4,0.6,0.8,1.])
        plt.scatter(xs, ys, color='tab:blue', linewidths=3)
        plt.scatter(xs[0], ys[0], color='tab:orange', linewidths=6)
        # draw arrow lines
        plt.quiver(xs, ys,dx,dy,color='tab:green', angles='xy', scale_units='xy', scale=1, alpha=0.5)
        plt.title("{} nodes, total length {:.2f}".format(self.graph_size, cost))
        if savepath is not None:
            plt.savefig(savepath)
        plt.show()

