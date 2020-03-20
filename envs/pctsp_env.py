# [Yunqiu Xu] [20200319] a PCTSP environment

import numpy as np
import pickle
import os
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

class PCTSPProblem(object):
    def __init__(self, graph_size=20, stochastic=False, penalty_factor=3, seed=1234):
        print("""
            In PCTSP, each node has not only an associated prize,
            but also an associated penalty. The goal is to collect at
            least a minimum total prize, while minimizing the total
            tour length plus the sum of penalties of unvisited nodes
            """)
        self.maxlen_dict = {20:2., 50:3., 100:4.}
        self.graph_size = graph_size
        assert (self.graph_size in self.maxlen_dict), "Supported size: {}".format(self.maxlen_dict.keys())
        self.max_length = self.maxlen_dict[self.graph_size]
        self.stochastic = stochastic # deterministic or stochastic
        print("Whether stochastic: {}".format(self.stochastic))
        self.penalty_max = self.max_length * penalty_factor * 1. / self.graph_size
        print("Max penalty: {:.2f}".format(self.penalty_max))
        self.seed = seed

    def generate_batch_data(self, batch_size=1):
        """
        Generate a batch of pstsp data: {depot, loc, penalty, deterministic_prize, stochastic_prize}
        """

        depot_data = np.random.uniform(size=(batch_size, 2))
        loc_data = np.random.uniform(size=(batch_size, self.graph_size, 2))
        penalty_data = np.random.uniform(size=(batch_size, self.graph_size)) * self.penalty_max
        # Take uniform prizes
        # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
        # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
        # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
        deterministic_prize_data = np.random.uniform(size=(batch_size, self.graph_size)) * 4. / self.graph_size
        # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
        # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
        # stochastic prize is only revealed once the node is visited
        # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
        stochastic_prize_data = np.random.uniform(size=(batch_size, self.graph_size)) * deterministic_prize_data * 2
        return {
                'depot': depot_data,                       
                'loc': loc_data,        
                'penalty': penalty_data,
                'deterministic_prize': deterministic_prize_data,
                'stochastic_prize': stochastic_prize_data
                }


    def generate_test_dataset(self, dataset_size=1000, foldername="test_dataset/"):
        """
        During testing, you can only run with batch_size 1!
        The dataset is a list containing dataset_size samples, each is with batch_size 1
        """
        det_or_sto = "sto" if self.stochastic else "det"
        filename = "{}pctsp{}_{}_{}_seed{}.pkl".format(foldername, 
                                                    self.graph_size, 
                                                    det_or_sto, 
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


# -------------------------------------

    def compute_cost(self, inputs, selected_idxs, get_vis_info=False):
        """
        :param inputs: a batch of pctsp data
        :param selected_idxs: a batch of selected idxs
        :return cost:
        """
        if selected_idxs.shape[-1] == 1:  
            # In case all tours directly return to depot, prevent further problems
            assert (selected_idxs == 0).all(), "If all length 1 tours, they should be zero"
            return np.zeros(selected_idxs.shape[0])

        # Compute prize
        prize_data = inputs['stochastic_prize'] if self.stochastic else inputs['deterministic_prize']
        prize_with_depot = np.concatenate([np.zeros_like(prize_data[:,:1]), prize_data], 1)
        prize = np.take_along_axis(prize_with_depot, selected_idxs, axis=1)
        total_prize = prize.sum(-1)

        # Compute penalty
        penalty_with_depot = np.concatenate([np.zeros_like(inputs['penalty'][:,:1]), inputs['penalty']], 1)
        penalty = np.take_along_axis(penalty_with_depot, selected_idxs, axis=1)
        unvisited_penalty = inputs['penalty'].sum(-1) - penalty.sum(-1)

        # Compute tour length
        loc_with_depot = np.concatenate([inputs['depot'][:, None, :], inputs['loc']], 1)
        idxs_to_gather = np.tile(selected_idxs[..., None], (loc_with_depot.shape[-1]))
        indexed_data = np.take_along_axis(loc_with_depot, idxs_to_gather, axis=1)
        length_part1 = np.linalg.norm(indexed_data[:,1:] - indexed_data[:,:-1], axis=2).sum(1) 
        length_part2 = np.linalg.norm(indexed_data[:,0] - inputs['depot'], axis=1) # Depot to first
        length_part3 = np.linalg.norm(indexed_data[:,-1] - inputs['depot'], axis=1) # Last to depot, will be 0 if depot is last
        total_length = length_part1 + length_part2 + length_part3

        # Compute cost. We want to maximize total prize but code minimizes so return negative
        # Incurred penalty cost is total penalty cost - saved penalty costs of nodes visited
        cost = total_length + unvisited_penalty

        if get_vis_info:
            assert(selected_idxs.shape[0] == 1), "Can only be used when batch_size == 1"
            return indexed_data[0], total_prize[0], unvisited_penalty[0], total_length[0], cost[0]
        return cost


    def plot(self, inputs, selected_idxs, 
            show_prize_and_penalty=True, savepath=None):
        depot = inputs['depot'][0]
        locs = inputs['loc'][0]
        if self.stochastic:
            prize = inputs['stochastic_prize'][0]
        else:
            prize = inputs['deterministic_prize'][0]
        penalty = inputs['penalty'][0]

        # Get indexed data
        indexed_data, total_prize, unvisited_penalty, total_length, cost = self.compute_cost(inputs, selected_idxs, get_vis_info=True)

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

        plt.plot(depot[0], depot[1], 'sk', color='tab:red', markersize=15)       # depot
        plt.scatter(locs[:,0], locs[:,1], color='tab:blue', linewidths=3)        # locs
        plt.scatter(xs, ys, color='tab:orange', linewidths=3)                    # selected nodes
        plt.quiver(xs, ys,dx,dy,color='tab:orange', angles='xy', scale_units='xy', scale=1, alpha=0.3) # route
        title = "Select {}/{}|Prize: {:.2f}|Length: {:.2f}|Penalty: {:.2f}|Cost: {:.2f}"
        title = title.format(selected_idxs.shape[1], 
                             self.graph_size,
                             total_prize,
                             total_length,
                             unvisited_penalty,
                             cost)
        plt.title(title)

        if show_prize_and_penalty:
            prize_rects = []
            penalty_rects = []
            for i in range(prize.shape[0]):
                curr_locs = locs[i]
                curr_prize = prize[i]
                curr_penalty = penalty[i]
                prize_rects.append(Rectangle((curr_locs[0], curr_locs[1]), 0.01, curr_prize))
                penalty_rects.append(Rectangle((curr_locs[0]+0.01, curr_locs[1]), 0.01, 0.5 * curr_penalty))
            
            pc_prize = PatchCollection(prize_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
            pc_penalty = PatchCollection(penalty_rects, facecolor='green', alpha=0.5, edgecolor='green')
            plt.axes().add_collection(pc_prize)
            plt.axes().add_collection(pc_penalty)

        if savepath is not None:
            plt.savefig(savepath)
        plt.show()
