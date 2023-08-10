from models.gnn_wrapper.NetWrapper import NetWrapper

from experiments.Experiment import Experiment
from models.utils.EarlyStopper import ClassificationMetrics
import torch

def get_sampleweight(labels):
    total = len(labels)
    class_sample_count = torch.unique(torch.tensor(labels), return_counts=True)[1]
    weight = class_sample_count.float()/total
    # need normalize.
    print('weight: ', weight, weight.shape)
    return weight

class EndToEndExperiment(Experiment):

    def __init__(self, model_configuration, exp_path):
        super(EndToEndExperiment, self).__init__(model_configuration, exp_path)


    def init_dataset(self):
        dataset_class = self.model_config.dataset  # dataset_class()
        
        if 'dense' in self.model_config:
            dataset = dataset_class(dense=self.model_config.dense, config=self.model_config)
        elif 'additional_features' in self.model_config:
            print('create node additional_features:', self.model_config.additional_features)
            dataset = dataset_class(additional_features = self.model_config.additional_features, 
                                    config=self.model_config)
        elif 'additional_graph_features' in self.model_config:
            print('create additional_graph_features', self.model_config.additional_graph_features)
            dataset = dataset_class(additional_graph_features = self.model_config.additional_graph_features, config=self.model_config)
        else:
            dataset = dataset_class(config=self.model_config)
            
        return dataset
        
    
    def run_valid(self, dataset_getter, logger, other=None) -> ClassificationMetrics:
        """
        This function returns the training and validation or test accuracy
        :return: (training accuracy, validation/test accuracy)
        """

        # print(self.model_config, dataset_getter.outer_k, dataset_getter.inner_k)

      
        model_class = self.model_config.model
        loss_class = self.model_config.loss
        optim_class = self.model_config.optimizer
        sched_class = self.model_config.scheduler
        stopper_class = self.model_config.early_stopper
        clipping = self.model_config.gradient_clipping

        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True
        dataset = self.init_dataset()
        
        train_loader, val_loader = dataset_getter.get_train_val(dataset, self.model_config['batch_size'],
                                                                shuffle=shuffle)
        
        print('dataset dim features', dataset.dim_features)
        print('dataset edge_attr_dim', dataset.edge_attr_dim)
        
        model = model_class(dim_features=dataset.dim_features, edge_attr_dim=dataset.edge_attr_dim, 
                            dim_target=dataset.dim_target, config=self.model_config)
        # NOTE: add weight
        labels = dataset.get_labels()
        # labels = [i.y for i in train_loader.dataset]
        # labels = train_loader.data.y
        # weight = get_sampleweight(labels)
        # weight = weight.to(self.model_config['device'])
        
        weight = None

        net = NetWrapper(model, loss_function=loss_class(weight=weight), device=self.model_config['device'], config=self.model_config)


        optimizer = optim_class(model.parameters(),
                                lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])

        if sched_class is not None:
            scheduler = sched_class(optimizer)
        else:
            scheduler = None
    
        metrics = net.train(train_loader=train_loader,
                                                    max_epochs=self.model_config['classifier_epochs'],
                                                    optimizer=optimizer, scheduler=scheduler,
                                                    clipping=clipping,
                                                    validation_loader=val_loader,
                                                    early_stopping=stopper_class,
                                                    logger=logger)
        return metrics

    def run_test(self, dataset_getter, logger, other=None) -> ClassificationMetrics:
        """
        This function returns the training and test accuracy. DO NOT USE THE TEST FOR TRAINING OR EARLY STOPPING!
        :return: (training accuracy, test accuracy)
        """

        dataset = self.init_dataset()
            
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True

        model_class = self.model_config.model
        loss_class = self.model_config.loss
        optim_class = self.model_config.optimizer
        sched_class = self.model_config.scheduler
        stopper_class = self.model_config.early_stopper
        clipping = self.model_config.gradient_clipping

        train_loader, val_loader = dataset_getter.get_train_val(dataset, self.model_config['batch_size'],
                                                                shuffle=shuffle)
        test_loader = dataset_getter.get_test(dataset, self.model_config['batch_size'], shuffle=shuffle)

        labels = [i.y for i in train_loader.dataset]
        # weight = get_sampleweight(labels)
        # weight = weight.to(self.model_config['device'])
        weight = None

        print('--------- self model config:', self.model_config)
        model = model_class(dim_features=dataset.dim_features, 
                            edge_attr_dim=dataset.edge_attr_dim,
                            dim_target=dataset.dim_target,
                            config=self.model_config)
        net = NetWrapper(model, loss_function=loss_class(weight=weight), device=self.model_config['device'], config=self.model_config)

        optimizer = optim_class(model.parameters(),
                                lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])

        if sched_class is not None:
            scheduler = sched_class(optimizer)
        else:
            scheduler = None

        metrics = net.train(train_loader=train_loader, max_epochs=self.model_config['classifier_epochs'],
                      optimizer=optimizer, scheduler=scheduler, clipping=clipping,
                      validation_loader=val_loader, test_loader=test_loader, early_stopping=stopper_class,
                      logger=logger)

        return metrics
