# [Yunqiu Xu] [20200319] an OP environment

import numpy as np
import pickle
import os
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

class OPProblem(object):
    def __init__(self, graph_size=20, prize_type='dist', seed=1234):
        self.maxlen_dict = {20:2., 50:3., 100:4.}
        self.graph_size = graph_size
        self.prize_type = prize_type # dist/const/unif
        assert (self.graph_size in self.maxlen_dict), "Supported size: {}".format(self.maxlen_dict.keys())
        print("Prize (dist|const|unif): {}".format(self.prize_type))
        self.max_length = self.maxlen_dict[self.graph_size]
        self.seed = seed

    def generate_batch_data(self, batch_size=1):
        """
        Generate a batch of op data: {depot, loc, prize, max_length}
        """
        depot_data = np.random.uniform(size=(batch_size, 2))
        loc_data = np.random.uniform(size=(batch_size, self.graph_size, 2))
        max_length_data = np.full(batch_size, self.max_length)
        # Methods taken from Fischetti et al. 1998
        if self.prize_type == 'const':
            prize_data = np.ones((batch_size, self.graph_size))
        elif self.prize_type == 'unif':
            prize_data = (1 + np.random.randint(0, 100, size=(batch_size, self.graph_size))) / 100.
        else:  # Based on distance to depot
            assert self.prize_type == 'dist'
            prize_ = np.linalg.norm(depot_data[:, None, :] - loc_data, axis=-1)
            prize_data = (1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.

        return {
                'depot': depot_data,                       
                'loc': loc_data,        
                'prize': prize_data,
                'max_length': max_length_data
                }


    def generate_test_dataset(self, dataset_size=1000, foldername="test_dataset/"):
        """
        During testing, you can only run with batch_size 1!
        The dataset is a list containing dataset_size samples, each is with batch_size 1
        """
        filename = "{}op{}_{}_{}_seed{}.pkl".format(foldername, 
                                                    self.graph_size, 
                                                    self.prize_type, 
                                                    dataset_size, 
                                                    self.seed)
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
        sample_data = dataset[0]
        print("Keys of sample data: {}".format(sample_data.keys()))
        for curr_key in sample_data.keys():
            print("{}: {}, {}".format(curr_key, sample_data[curr_key].shape, sample_data[curr_key].dtype))
        return dataset


    def compute_cost(self, inputs, selected_idxs, get_vis_info=False):
        """
        :param inputs: a batch of op data
        :param selected_idxs: a batch of selected idxs
        :return cost:
        """
        if selected_idxs.shape[-1] == 1:  
            # In case all tours directly return to depot, prevent further problems
            assert (selected_idxs == 0).all(), "If all length 1 tours, they should be zero"
            return np.zeros(selected_idxs.shape[0])

        # Compute tour length, but only for assertion
        loc_with_depot = np.concatenate([inputs['depot'][:, None, :], inputs['loc']], 1)
        idxs_to_gather = np.tile(selected_idxs[..., None], (loc_with_depot.shape[-1]))
        indexed_data = np.take_along_axis(loc_with_depot, idxs_to_gather, axis=1)
        length_part1 = np.linalg.norm(indexed_data[:,1:] - indexed_data[:,:-1], axis=2).sum(1) 
        length_part2 = np.linalg.norm(indexed_data[:,0] - inputs['depot'], axis=1) # Depot to first
        length_part3 = np.linalg.norm(indexed_data[:,-1] - inputs['depot'], axis=1) # Last to depot, will be 0 if depot is last
        total_length = length_part1 + length_part2 + length_part3
        assert (total_length <= self.max_length + 1e-5).all(), \
            "Max length exceeded by {:.2f}".format((total_length - self.max_length).max())

        # Compute prize
        prize_with_depot = np.concatenate([np.zeros_like(inputs['prize'][:,:1]), inputs['prize']], 1)
        prize = np.take_along_axis(prize_with_depot, selected_idxs, axis=1)
        cost = -prize.sum(-1)
        if get_vis_info:
            assert(selected_idxs.shape[0] == 1), "Can only be used when batch_size == 1"
            return indexed_data[0], total_length[0], -cost[0]
        return cost


    def plot(self, inputs, selected_idxs, 
                    show_prize=True, 
                    savepath=None):
        depot = inputs['depot'][0]
        locs = inputs['loc'][0]
        prize = inputs['prize'][0]
        max_length = inputs['max_length'][0]

        # Get indexed data
        indexed_data, total_length, total_prize = self.compute_cost(inputs, selected_idxs, get_vis_info=True)

        # Visualization
        xs = indexed_data[:,0]
        ys = indexed_data[:,1]
        dx = np.roll(xs, -1) - xs
        dy = np.roll(ys, -1) - ys

        plt.figure(figsize=(10,10))
        plt.xlim(0,1)
        plt.xticks([0.,0.2,0.4,0.6,0.8,1.])
        plt.ylim(0,1)
        plt.yticks([0.,0.2,0.4,0.6,0.8,1.])

        plt.plot(depot[0], depot[1], 'sk', color='tab:red', markersize=15)
        plt.scatter(locs[:,0], locs[:,1], color='tab:blue', linewidths=3)
        plt.scatter(xs, ys, color='tab:orange', linewidths=3)
        plt.quiver(xs, ys,dx,dy,color='tab:orange', angles='xy', scale_units='xy', scale=1, alpha=0.3)
        plt.title("Select {}/{}|Length {:.2f}/{:.2f}|Prize {:.2f}".format(selected_idxs.shape[1],
                                                                          self.graph_size,
                                                                          total_length,
                                                                          self.max_length,
                                                                          total_prize))

        if show_prize:
            prize_rects = []
            for i in range(prize.shape[0]):
                curr_locs = locs[i]
                curr_prize = prize[i]
                prize_rects.append(Rectangle((curr_locs[0], curr_locs[1]), 0.01, 0.1 * curr_prize))
            pc_prize = PatchCollection(prize_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
            plt.axes().add_collection(pc_prize)

        if savepath is not None:
            plt.savefig(savepath)
        plt.show()
