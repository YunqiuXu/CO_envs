# [Yunqiu Xu] [20200318] a cvrp environment

import numpy as np
import pickle
import os
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

class CVRPProblem(object):
    def __init__(self, graph_size=20, seed=1234):
        self.capacity_dict = {10:20., 20:30., 50:40., 100:50.}
        self.graph_size = graph_size
        assert (self.graph_size in self.capacity_dict), "Supported size: {}".format(self.capacity_dict.keys())
        self.capacity_size = self.capacity_dict[self.graph_size]
        self.seed = seed

    def generate_vrp_data(self, batch_size=1):
        """
        Generate a batch of vrp data: {depot, loc, demand}
        
        depot: the location of depot, the vehicle starts here and will return here if
            1) the capacity is full
            2) all nodes have been visited
        loc: the location of nodes, the vehicle need to visit each node
        demand: integer 1-9, scaled by the capacity
        """
        return {
                'depot': np.random.uniform(size=(batch_size, 2)),                       
                'loc': np.random.uniform(size=(batch_size, self.graph_size, 2)),        
                'demand': np.random.randint(1, 10, size=(batch_size, self.graph_size)) * 1. / self.capacity_size,
                }


    def generate_test_dataset(self, dataset_size=1000, foldername="xyq_test_dataset/"):
        """
        During testing, you can only run with batch_size 1!
        The dataset is a list containing dataset_size samples, each is with batch_size 1
        """
        filename = "{}cvrp{}_{}_seed{}.pkl".format(foldername, self.graph_size, dataset_size, self.seed)
        print("Generate test dataset at: {}".format(filename))
        np.random.seed(self.seed)
        dataset = []
        for i in range(dataset_size):
            curr_data = self.generate_vrp_data(batch_size=1)
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
        print("depot: {}, {}".format(sample_data['depot'].shape, sample_data['depot'].dtype))
        print("loc: {}, {}".format(sample_data['loc'].shape, sample_data['loc'].dtype))
        print("demand: {}, {}".format(sample_data['demand'].shape, sample_data['demand'].dtype))
        return dataset


    def compute_cost(self, inputs, selected_idxs):
        """
        :param inputs: a batch of cvrp data {depot:(batch_size, 2), 
                                            loc:(batch_size, graph_size, 2), 
                                            demand:(batch_size, graph_size)
        :param selected_idxs: a batch of selected idxs, (batch_size, tour_length), since the depot will be 
                            visited multiple times, the tour_length > graph_size
        :return cost: (batch_size,)
        """

        # concatenate loc with depot: (batch_size, graph_size+1, 2)
        loc_with_depot = np.concatenate([inputs['depot'][:, None, :], inputs['loc']], 1)        
        # Get indexed data, the gather-like operation is taken from 
        #   https://stackoverflow.com/questions/46868056/how-to-gather-elements-of-specific-indices-in-numpy
        # selected_idxs[..., None]: 
        # after expanding: (batch_size, tour_length, 2)
        # after gathering: 
        # (batch_size, tour_length) -> (batch_size, tour_length, 1) -> (batch_size, tour_length, 2)
        # idxs_to_gather = selected_idxs[..., None].expand(*selected_idxs.shape, loc_with_depot.shape[-1])
        idxs_to_gather = np.tile(selected_idxs[..., None], (loc_with_depot.shape[-1]))
        indexed_data = np.take_along_axis(loc_with_depot, idxs_to_gather, axis=1)
        # Compute cost
        cost_part1 = np.linalg.norm(indexed_data[:,1:] - indexed_data[:,:-1], axis=2).sum(1) 
        cost_part2 = np.linalg.norm(indexed_data[:,0] - inputs['depot'], axis=1) # Depot to first
        cost_part3 = np.linalg.norm(indexed_data[:,-1] - inputs['depot'], axis=1) # Last to depot, will be 0 if depot is last
        return cost_part1 + cost_part2 + cost_part3


    def plot_cvrp(self, inputs, selected_idxs, 
                    round_demand=False, 
                    visualize_demands=False, 
                    savepath=None):
        """
        savepath = "xyq_vis/cvrp20_fig.png"
        """
        def discrete_cmap(N, base_cmap=None):
            """
            Create an N-bin discrete colormap from the specified input map
            """
            # Note that if base_cmap is a string or None, you can simply do
            #    return plt.cm.get_cmap(base_cmap, N)
            # The following works for string, None, or a colormap instance:
            base = plt.cm.get_cmap(base_cmap)
            color_list = base(np.linspace(0, 1, N))
            cmap_name = base.name + str(N)
            return base.from_list(cmap_name, color_list, N)

        assert (selected_idxs.shape[0] == 1), "For visualization, batch_size == 1!"
        selected_idxs = selected_idxs.squeeze(0)
        # separating different routes with 0 (depot)
        routes = [r[r!=0] for r in np.split(selected_idxs, np.where(selected_idxs==0)[0]) if (r != 0).any()]
        depot = inputs['depot'][0]
        locs = inputs['loc'][0]
        demands = inputs['demand'][0]
        capacity = self.capacity_dict[self.graph_size] # e.g. for cvrp100, the demand_scale is 50

        # setup figure
        plt.figure(figsize=(10,10))
        plt.xlim(0,1)
        plt.xticks([0.,0.2,0.4,0.6,0.8,1.])
        plt.ylim(0,1)
        plt.yticks([0.,0.2,0.4,0.6,0.8,1.])
        markersize = 5

        # visualize
        x_dep, y_dep = depot
        print("Depot: ({:.2f},{:.2f})".format(x_dep, y_dep))
        plt.plot(x_dep, y_dep, 'sk', markersize=markersize*4)

        cmap = discrete_cmap(len(routes) + 2, 'nipy_spectral')
        dem_rects = []
        used_rects = []
        cap_rects = []
        qvs = []
        total_dist = 0
        for veh_number, r in enumerate(routes):
            color = cmap(len(routes) - veh_number) # Invert to have in rainbow order
            route_demands = demands[r - 1]
            coords = locs[r - 1, :]
            xs, ys = coords.transpose()
            total_route_demand = sum(route_demands)
            assert total_route_demand <= capacity
            if not visualize_demands:
                plt.plot(xs, ys, 'o', mfc=color, markersize=markersize, markeredgewidth=0.0)
            dist = 0
            x_prev, y_prev = x_dep, y_dep
            cum_demand = 0
            for (x, y), d in zip(coords, route_demands):
                dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)
                cap_rects.append(Rectangle((x, y), 0.01, 0.1))
                used_rects.append(Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity))
                dem_rects.append(Rectangle((x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity))
                x_prev, y_prev = x, y
                cum_demand += d
            dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
            total_dist += dist
            qv = plt.quiver(xs[:-1], ys[:-1], xs[1:] - xs[:-1], ys[1:] - ys[:-1],
                            scale_units='xy', angles='xy', scale=1, color=color,
                            label='R{}, # {}, c {} / {}, d {:.2f}'.format(
                                veh_number, 
                                len(r), 
                                int(total_route_demand) if round_demand else total_route_demand, 
                                int(capacity) if round_demand else capacity,
                                dist)
                            )
            qvs.append(qv)
        plt.legend(handles = qvs, loc='upper center')
        plt.title("CVRP{}, {} routes, total distance {:.2f}".format(self.graph_size, len(routes), total_dist))
        pc_cap = PatchCollection(cap_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
        pc_used = PatchCollection(used_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray')
        pc_dem = PatchCollection(dem_rects, facecolor='black', alpha=1.0, edgecolor='black')
        if visualize_demands:
            plt.axes().add_collection(pc_cap)
            plt.axes().add_collection(pc_used)
            plt.axes().add_collection(pc_dem)
        if savepath is not None:
            plt.savefig(savepath)
        plt.show()
