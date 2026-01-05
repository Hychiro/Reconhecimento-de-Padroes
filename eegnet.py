import numpy as np
import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
import numpy as np
import torch.nn.functional as F

class LabelsMapper:
    def __init__(self, original_array):
        self.original_array = original_array
        self.label_map_dict, self.mapped_array = self.map_array()

    def map_array(self):
        unique_values = list(set(self.original_array))
        class_mapping = {val: idx for idx, val in enumerate(unique_values)}
        classified_arr = [class_mapping[val] for val in self.original_array]
        return class_mapping, classified_arr

    def map_array_using_dict(self, array):
        mapped_array = [self.label_map_dict[val] if val in self.label_map_dict else val for val in array]
        return mapped_array



def get_minibatches(X, y, batchsize):
    ''' Generates mini-batches from the input data.

    Description
    -----------
    This function generates mini-batches from the input data X and labels y. It shuffles the data
    and returns a subset of the specified batch size.

    Parameters
    ----------
    X : array-like, shape (n_samples, *n_features)
        The input features.
    y : array-like, shape (n_samples,)
        The labels.
    batchsize : int
        The size of the mini-batch.

    Returns
    -------
    X_batch : array-like, shape (batchsize, n_features)
        The input features for the mini-batch.
    y_batch : array-like, shape (batchsize,)
        The labels for the mini-batch.
    '''
    batch = np.array([i for i in range(len(X))])
    np.random.shuffle(batch)
    return X[batch[:batchsize]], y[batch[:batchsize]]


def setup_tensors(X, y=None, device="cpu"):
    ''' Sets up the input features and labels as tensors.

    Description
    -----------
    This function converts the input features and labels to PyTorch tensors, the correct dtype, and moves them to
    the specified device.

    Parameters
    ----------
    X : array-like, shape (n_samples, *n_features)
        The input features.
    y : array-like, shape (n_samples,), optional
        The labels.
    device : str, optional
        The device to move the tensors to (default is 'cpu').

    Returns
    -------
    X_tensor : torch.Tensor
        The input features as a tensor.
    y_tensor : torch.Tensor, optional
        The labels as a tensor.
    '''
    X_temp_tensor = torch.tensor(np.array(X).astype(np.float32)).to(dtype=torch.float32, device=device)

    if y is not None:
        y_temp_tensor = torch.tensor(np.array(y).astype(int)).to(dtype=torch.long, device=device)
    else:
        y_temp_tensor = None

    return X_temp_tensor, y_temp_tensor


class SklearnStructure:
    ''' A base class that mimics the structure of scikit-learn classifiers.

    Description
    -----------
    This class provides a base structure for PyTorch models that mimic the interface of
    scikit-learn classifiers. It includes methods for prediction and model training.

    Attributes
    ----------
    device : str
        The device to use for computations.

    Methods
    -------
    predict_proba(X):
        Predicts the probability of each class given the features.
    predict(X):
        Predicts the class of the input features.
    get_model_size():
        Prints a message indicating the function is missing.
    train_model(X, y, batchsize=32, lr=0.001, iterations=2000, verbose=False):
        Trains the model on the input features and labels.
    '''
    def __init__(self, device="cpu"):
        ''' Initializes the class.

        Parameters
        ----------
        device : str, optional
            The device to use for computations (default is 'cpu').
        '''
        self.device = device

    def predict_proba(self, X):
        ''' Predicts the probability of each class given the features.

        Parameters
        ----------
        X : array-like, shape (n_samples, *n_features)
            The input features.

        Returns
        -------
        proba : array-like, shape (n_samples, n_classes)
            The probabilities of each class for each input feature.
        '''
        self.model.eval()
        self.model.to("cpu")
        X_tensor, _ = setup_tensors(X, device=self.device)

        pred, loss = self.model(X_tensor, None)

        self.model.train()

        return F.softmax(pred, dim=-1).detach().cpu().numpy()

    def predict(self, X):
        ''' Predicts the class of the input features.

        Parameters
        ----------
        X : array-like, shape (n_samples, *n_features)
            The input features.

        Returns
        -------
        pred : array-like, shape (n_samples,)
            The predicted classes for the input features.
        '''
        return np.argmax(self.predict_proba(X), axis=-1)

    def get_model_size(self):
        ''' Prints a message indicating the function is missing.

        Returns
        -------
        int
            Returns 0.
        '''
        print('Missing Function')
        return 0

    def train_model(self, X, y, batchsize=32, lr=0.001, iterations=2000, verbose=False):
        ''' Trains the model on the input features and labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, *n_features)
            The input features.
        y : array-like, shape (n_samples,)
            The labels.
        batchsize : int, optional
            The size of the mini-batch (default is 32).
        lr : float, optional
            The learning rate (default is 0.001).
        iterations : int, optional
            The number of iterations (default is 2000).
        verbose : bool, optional
            If True, prints the progress of the training process (default is False).
        '''
        self.model.to(device=self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        X_tensor, y_tensor = setup_tensors(X, y, self.device)

        for it in range(iterations):
            X_temp, y_temp = get_minibatches(X_tensor, y_tensor, batchsize)
            pred, loss = self.model(X_temp, y_temp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('loss: %.4f, iter: %d' % (loss, it), end='\r')


class Conv2dWithConstraint(nn.Conv2d):
    ''' A 2D convolutional layer with weight normalization constraint.

    Description
    -----------
    This class implements a 2D convolutional layer with a constraint on the maximum norm
    of the weights.

    Attributes
    ----------
    max_norm : float
        The maximum norm of the weights.

    Methods
    -------
    forward(x):
        Applies the convolutional layer to the input tensor.
    '''
    def __init__(self, *config, max_norm=1, **kwconfig):
        ''' Initializes the class.

        Parameters
        ----------
        *config : tuple
            Configuration parameters for the convolutional layer.
        max_norm : float, optional
            The maximum norm of the weights (default is 1).
        **kwconfig : dict
            Additional configuration parameters for the convolutional layer.
        '''
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        ''' Applies the convolutional layer to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        '''
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    ''' A linear layer with weight normalization constraint.

    Description
    -----------
    This class implements a linear layer with a constraint on the maximum norm
    of the weights.

    Attributes
    ----------
    max_norm : float
        The maximum norm of the weights.

    Methods
    -------
    forward(x):
        Applies the linear layer to the input tensor.
    '''
    def __init__(self, *config, max_norm=1, **kwconfig):
        ''' Initializes the class.

        Parameters
        ----------
        *config : tuple
            Configuration parameters for the linear layer.
        max_norm : float, optional
            The maximum norm of the weights (default is 1).
        **kwconfig : dict
            Additional configuration parameters for the linear layer.
        '''
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        ''' Applies the linear layer to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        '''
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)


class EEGNetBlock(nn.Module):
    ''' A building block for the EEGNet model.

    Description
    -----------
    This class implements a building block for the EEGNet model, consisting of
    temporal, spatial, and separable convolutional layers.

    Attributes
    ----------
    n_convs : list
        The number of convolutional filters for each layer.
    temporal_conv_size : int
        The size of the temporal convolutional filter.
    temporal_conv_padding : str
        The padding type for the temporal convolutional layer.
    separable_conv_size : int
        The size of the separable convolutional filter.
    separable_conv_padding : str
        The padding type for the separable convolutional layer.
    polling_size : list
        The size of the pooling layers.
    pooling_kind : list
        The type of pooling layers ('avg' or 'max').
    dropout_rate : float
        The dropout rate.
    n_channels : int
        The number of input channels.
    n_times : int
        The number of time steps in the input.
    filters : int
        The number of filters.

    Methods
    -------
    forward(x):
        Applies the EEGNet block to the input tensor.
    '''
    def __init__(self,
                 n_convs=[8, 2, 16],
                 temporal_conv_size=64, temporal_conv_padding='same',
                 separable_conv_size=16, separable_conv_padding='same',
                 polling_size=[4, 8], pooling_kind=['avg', 'avg'], dropout_rate=0.5,
                 n_channels=22, n_times=256, filters=1):

        ''' Initializes the class.

        Parameters
        ----------
        n_convs : list, optional
            The number of convolutional filters for each layer (default is [8, 2, 16]).
        temporal_conv_size : int, optional
            The size of the temporal convolutional filter (default is 64).
        temporal_conv_padding : str, optional
            The padding type for the temporal convolutional layer (default is 'same').
        separable_conv_size : int, optional
            The size of the separable convolutional filter (default is 16).
        separable_conv_padding : str, optional
            The padding type for the separable convolutional layer (default is 'same').
        polling_size : list, optional
            The size of the pooling layers (default is [4, 8]).
        pooling_kind : list, optional
            The type of pooling layers ('avg' or 'max') (default is ['avg', 'avg']).
        dropout_rate : float, optional
            The dropout rate (default is 0.5).
        n_channels : int, optional
            The number of input channels (default is 22).
        n_times : int, optional
            The number of time steps in the input (default is 256).
        filters : int, optional
            The number of filters (default is 1).
        '''
        super().__init__()
        pooling_kind_dict = {'avg': nn.AvgPool2d, 'max': nn.MaxPool2d}
        self.n_features = n_times
        temporal_conv_size = n_times // 4 if temporal_conv_size == 'auto' else temporal_conv_size

        # Temporal Conv
        if filters > 1:
            rearrange = Rearrange('n f c t -> n f c t')
        else:
            rearrange = Rearrange('b c t -> b 1 c t')
        conv1 = nn.Conv2d(filters, n_convs[0], (1, temporal_conv_size), bias=False, padding=temporal_conv_padding)
        batch1 = nn.BatchNorm2d(n_convs[0])

        self.n_features = self.n_features - temporal_conv_size + 1 if temporal_conv_padding == 'valid' else self.n_features

        # Spatial Conv
        conv2 = Conv2dWithConstraint(n_convs[0], n_convs[0] * n_convs[1],
                                     (n_channels, 1), bias=False, groups=n_convs[0], max_norm=1)
        batch2 = nn.BatchNorm2d(n_convs[0] * n_convs[1])
        activation1 = nn.ReLU()
        pool1 = pooling_kind_dict[pooling_kind[0]]((1, polling_size[0]))
        dropout1 = nn.Dropout(dropout_rate)

        self.n_features = self.n_features // polling_size[0]

        # Separable Conv
        separable_conv_size = self.n_features // 4 if separable_conv_size == 'auto' else separable_conv_size
        conv3 = nn.Conv2d(n_convs[0] * n_convs[1], n_convs[0] * n_convs[1], (1, separable_conv_size), bias=False,
                          padding=separable_conv_padding, groups=n_convs[0] * n_convs[1])
        conv4 = nn.Conv2d(n_convs[0] * n_convs[1], n_convs[2], 1, bias=False)
        batch3 = nn.BatchNorm2d(n_convs[2])
        activation2 = nn.ReLU()
        pool2 = pooling_kind_dict[pooling_kind[1]]((1, polling_size[1]))
        dropout2 = nn.Dropout(dropout_rate)

        self.n_features = self.n_features - separable_conv_size + 1 if separable_conv_padding == 'valid' else self.n_features
        self.n_features = self.n_features // polling_size[1]

        self.n_features = n_convs[2] * self.n_features

        temporal = nn.Sequential(rearrange, conv1, batch1)
        spatial = nn.Sequential(conv2, batch2, activation1, pool1, dropout1)
        separable = nn.Sequential(conv3, conv4, batch3, activation2, pool2, dropout2)

        self.net = nn.Sequential(temporal, spatial, separable)

    def forward(self, x):
        ''' Applies the EEGNet block to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        '''
        return self.net(x)


class MBEEGNetModel(nn.Module):
    ''' A model that combines multiple EEGNet blocks.

    Description
    -----------
    This class implements a model that combines multiple EEGNet blocks, each for one signal band. 
    Then the EEGNet blocks outputs are stacked and go through a linear layer for classification.

    Attributes
    ----------
    EEGNetBlocks : nn.ModuleList
        The list of EEGNet blocks.
    head : nn.Sequential
        The linear layer for classification.

    Methods
    -------
    forward(x, targets):
        Applies the model to the input tensor.
    '''
    def __init__(self, bands, classes, EEGNet_args):
        ''' Initializes the class.

        Parameters
        ----------
        bands : int
            The number of EEG bands.
        classes : int
            The number of classes.
        EEGNet_args : list
            The arguments for the EEGNet block.
        '''
        super().__init__()
        self.EEGNetBlocks = nn.ModuleList([EEGNetBlock(*EEGNet_args) for i in range(bands)])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sum([e.n_features for e in self.EEGNetBlocks]), classes)
        )

    def forward(self, x, targets):
        ''' Applies the model to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        targets : torch.Tensor, optional
            The target labels.

        Returns
        -------
        logits : torch.Tensor
            The output logits.
        loss : torch.Tensor, optional
            The cross-entropy loss if targets are provided.
        '''
        results = []
        for i in range(x.shape[1]):
            results.append(self.EEGNetBlocks[i](x[:, i, :, :]))
        results = torch.stack(results, dim=1)

        logits = self.head(results)

        if targets is None:
            loss = None
        else:
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss


class MBEEGNet(SklearnStructure):
    ''' An adapted EEGNet scikit-learn-like model for multi-band signals.

    Description
    -----------
    This class implements a model for multi-band EEGNet, inheriting from SklearnStructure
    to provide scikit-learn-like interface.

    Attributes
    ----------
    n_convs : list
        The number of convolutional filters for each layer.
    temporal_conv_size : int
        The size of the temporal convolutional filter.
    temporal_conv_padding : str
        The padding type for the temporal convolutional layer.
    separable_conv_size : int
        The size of the separable convolutional filter.
    separable_conv_padding : str
        The padding type for the separable convolutional layer.
    polling_size : list
        The size of the pooling layers.
    pooling_kind : list
        The type of pooling layers ('avg' or 'max').
    dropout_rate : float
        The dropout rate.
    device : str
        The device to use for computations.
    random_state : int
        The random seed.

    Methods
    -------
    fit(X, y, lr=0.001, iterations=1000, batchsize=64, verbose=False):
        Fits the model to the input features and labels.
    '''
    def __init__(self,
                 n_convs=[8, 2, 16],
                 temporal_conv_size=64, temporal_conv_padding='same',
                 separable_conv_size=16, separable_conv_padding='same',
                 polling_size=[4, 8], pooling_kind=['avg', 'avg'], dropout_rate=0.5,
                 device="cpu", random_state=42):
        ''' Initializes the class.

        Parameters
        ----------
        n_convs : list, optional
            The number of convolutional filters for each layer (default is [8, 2, 16]).
        temporal_conv_size : int, optional
            The size of the temporal convolutional filter (default is 64).
        temporal_conv_padding : str, optional
            The padding type for the temporal convolutional layer (default is 'same').
        separable_conv_size : int, optional
            The size of the separable convolutional filter (default is 16).
        separable_conv_padding : str, optional
            The padding type for the separable convolutional layer (default is 'same').
        polling_size : list, optional
            The size of the pooling layers (default is [4, 8]).
        pooling_kind : list, optional
            The type of pooling layers ('avg' or 'max') (default is ['avg', 'avg']).
        dropout_rate : float, optional
            The dropout rate (default is 0.5).
        device : str, optional
            The device to use for computations (default is 'cpu').
        random_state : int, optional
            The random seed (default is 42).
        '''
        super().__init__(device=device)

        self.device = device
        self.n_convs = n_convs
        self.temporal_conv_size = temporal_conv_size
        self.temporal_conv_padding = temporal_conv_padding
        self.separable_conv_size = separable_conv_size
        self.separable_conv_padding = separable_conv_padding
        self.polling_size = polling_size
        self.pooling_kind = pooling_kind
        self.dropout_rate = dropout_rate
        torch.manual_seed(random_state)

    def fit(self, X, y, lr=0.001, iterations=1000, batchsize=64, verbose=False):
        ''' Fits the model to the input features and labels.

            Description
            -----------
            This method initializes the model with the given parameters and trains it using the provided
            features and labels. It maps the labels, sets up the model, and calls the `train_model` method
            to perform the training.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_bands, n_channels, n_times)
                The input features. This should be a 4D array where `n_samples` is the number of samples,
                `n_bands` is the number of EEG bands, `n_channels` is the number of channels, and `n_times`
                is the number of time steps.

            y : array-like, shape (n_samples,)
                The labels corresponding to the input features. These labels will be mapped to a numerical
                format.

            lr : float, optional, default=0.001
                The learning rate for the optimizer.

            iterations : int, optional, default=1000
                The number of training iterations (epochs) to perform.

            batchsize : int, optional, default=64
                The size of the mini-batch used during training.

            verbose : bool, optional, default=False
                If True, prints progress messages during training.

            Returns
            -------
            self : object
                Returns the instance of the fitted model.

            Notes
            -----
            - The method uses `LabelsMapper` to convert the labels `y` into a numerical format suitable for
              training.
            - The model is initialized with parameters for the EEGNet blocks and trained using the `train_model`
              method.
        '''
        self.LabelsMapper = LabelsMapper(y)
        self.mapped_y = np.array(self.LabelsMapper.mapped_array)
        self.model = MBEEGNetModel(
            bands=X.shape[1],
            classes=len(self.LabelsMapper.label_map_dict),
            EEGNet_args=[
                self.n_convs,
                self.temporal_conv_size,
                self.temporal_conv_padding,
                self.separable_conv_size,
                self.separable_conv_padding,
                self.polling_size,
                self.pooling_kind,
                self.dropout_rate,
                X.shape[2],
                X.shape[-1],
                1,
            ]
        )

        self.train_model(X, self.mapped_y, batchsize=batchsize, lr=lr, iterations=iterations, verbose=verbose)

        return self

