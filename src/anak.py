from parser import parameter_parser


import torch_geometric.nn import GCNConv
import json
import tqdm
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F


class ListModule(nn.Module):
    """
    Add arbritary length and type of layers
    """
    def __init__(self,*args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx +=1

    def __getitem__(self, idx):
        if idx <0 or idx >= len(self.modules):
            raise IndexError('index {} is out of range',format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self.modules)


class PrimaryCapsuleLayer(nn.Modules):
    def __init__(self, in_units, in_channels, num_units, capsule_dimensions):

class CapsGNN(nn.Module):
    def __init__(self, args, number_of_features, number_of_targets):
        super(CapsGNN, self).__init__()

        self.args = args
        self.number_of_features = number_of_features
        self.number_of_targets = number_of_targets
        self._setup_layers()

    def _setup_base_layers(self):
        self.base_layers = [GCNConv(self.number_of_features, self.args.gcn_filters)]
        for layer in range(self.args.gcn_layers-1):
            self.base_layers.append(GCNConv(self.args.gcn_filters, self.args.gcn_filters))

        self.base_layers = ListModule(*self.base_layers)

    def _setup_primary_capsules(self):
        # todo 4
        self.first_capsules = PrimaryCapsuleLayer(in_units = self.args.gcn_filters,
                                                  in_channels = self.gcn_layers,
                                                  num_units = self.gcn_layer,
                                                  capsule_dimensions = self.args.capsule_dimensions)

    def _setup_layers(self):
        self._setup_bsse_laysers()
        self._setup_primary_cpasules()




def create_numeric_mapping(self, node_properties):
    return {value:i for i, value in enumerate(node_properties)}

class CapsGNNTrainer(object):

    def __init__(self, args):
        self.args= args
        self.setup_model()

    def enumerate_uniq_labels_nd_targets(self): # todo 2??
        """
        Enumerating the features and targets in order to setup weights later
        :return:
        """
        ending = "*json"

        self.train_graph_paths = glob.glob(self.args.train_graph_foldre+ending)
        self.test_graph_paths = glob.glob(self.args.test_graph_folder+ending)

        graph_paths = self.train_grpah_paths + self.test_grpah_paths

        targets = set()
        features = set()
        for path in tqdm(graph_paths):
            data = json.load(open(path))
            target = targets.union(set([data["target"]]))
            features = features.union(set(data['labels']))

        self.target_map = create_numeric_mapping(targets)
        self.feature_map = create_numeric_mapping(features)

        self.number_of_features = len(self.feature_map)
        self.number_of_targets = len(self.target_map)

    def setup_model(self):
        """Enumerating labels and initializing a CapsGNN"""
        self.enumerate_unique_labels_and_targets() # todo 1??
        self.model = CapsGNN(self.args, self.number_of_features, self.number_targets)




def main():
    """
    Parseing command line parameters, processing graphs, fitting a CapsGNN.
    """
    args = parameter_paraser()
    tab_printer(args)
    model = CapsGNNTrainer(args)

if __name__ =='__main__':
    main()
