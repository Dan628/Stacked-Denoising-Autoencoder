
"""Module containing stacked denoising autoencoder."""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, initializers, optimizers
from sklearn.utils.validation import _deprecate_positional_args
from .validation import InputValidation
from .visualize import Plot

@_deprecate_positional_args
class StackedDenoisingAutoencoder(InputValidation, Plot):
    """Stacked Denoising Autoencoder for performing feature extraction
    on tabular data.

    Attributes:
        n_features (int):
            The number of features in the original dataset
        target_size (int):
            the number of features we wish to extract
        epochs (int):
            The number of epochs to use in training the SDNAE
        learning_rate(float):
            Learning rate used to adjust gradients; default = 0.01
        input_noise (floatin):
            Range [0,1] represents the percent of zero masking applied
            using dropout during training on the input and output of model; default = 0.4
        inner_noise (float):
            Range [0,1] percent of dropout to apply on all
            middle layers; default 0.3
        output_noise(float):
            Range [0,1] percent of dropout to apply before passing
            to outout_layer; default 0.4
        loss (string):
            loss function used for training accepts any standard
            keras loss function; default = 'mean_squared_error'
        batch_size (int):
            Batch size to use in training; default = 32
        custom_layers(list of integers):
            list representing the number of neurons in each layer
            given as a list of seven elements where the first and last
            element must equal n_features and the third element must
            equal target_size; Default will linearly decrease layer sizes
            from input size to target_size for encoder and linearly increase
            layer size from target_size to n_features for decoder.
    """

    def __init__(self, n_features, target_size, epochs, *, learning_rate=0.01,
                 input_noise=0.4, inner_noise=0.3, output_noise=0.4,
                 loss='mean_squared_error', batch_size=32, custom_layers=None):

        self.n_features = n_features
        self.target_size = target_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.noise = input_noise
        self.inner_noise = inner_noise
        self.output_noise = output_noise
        self.loss = loss
        self.batch_size = batch_size
        self.custom_layers = custom_layers
        self.model = None

    def _establish_layers(self):
        """Method to establish default layer sizes. If custom layers
        is passed, this method calls check_custom_layers()

        Args:
            None

        Returns:
            None
        """

        first_reduction = ((self.n_features - self.target_size) / 3)
        second_reduction = first_reduction * 2

        if self.custom_layers is None:

        #create default layer size list
            self.custom_layers = [0] * 7
            self.custom_layers[0] = self.n_features
            self.custom_layers[1] = self.n_features - first_reduction
            self.custom_layers[2] = self.n_features - second_reduction
            self.custom_layers[3] = self.target_size
            self.custom_layers[4] = self.n_features - second_reduction
            self.custom_layers[5] = self.n_features - first_reduction
            self.custom_layers[6] = self.n_features

        else:
            self.check_custom_layers()

    def fit(self, train_data, verbose=0):
        """ The fit method assembles model, trains the model and saves the
        model to the self.model attribute using keras sequential class

        Args:
            train_data (array-like):
                Data for which to train model
            verbose (0,1,2):
                Keras verbosity mode used during training
                0 = silent, 1 = progress bar, 2 = one line per epoch.

        Returns:
            None
        """

        #Check Input Data
        train_data = self.check_data(train_data)

        #Establish and check self.custom_layers
        self._establish_layers()

        #Assemble model
        model = Sequential()
        model.add(layers.Dropout(self.noise, input_shape=(self.custom_layers[0],)))
        model.add(Dense(self.custom_layers[1], activation='relu', 
                        kernel_initializer='glorot_normal'))
        model.add(layers.Dropout(self.inner_noise))
        model.add(Dense(self.custom_layers[2], activation='relu', 
                        kernel_initializer='glorot_normal'))
        model.add(layers.Dropout(self.inner_noise))
        model.add(Dense(self.custom_layers[3], activation='linear', name="middle_layer", 
                        kernel_initializer='glorot_normal'))
        model.add(Dense(self.custom_layers[4], activation='relu'))
        model.add(layers.Dropout(self.inner_noise))
        model.add(Dense(self.custom_layers[5], activation='relu', 
                        kernel_initializer='glorot_normal'))
        model.add(layers.Dropout(self.output_noise))
        model.add(Dense(self.custom_layers[6], activation='sigmoid', 
                        kernel_initializer='glorot_normal')) 

        #Compile Model
        model.compile(loss=self.loss, optimizer=Adam(self.learning_rate))

        #Train Model
        model.fit(train_data, train_data, batch_size=self.batch_size, 
                  epochs=self.epochs, verbose=verbose)

        #Assign Model to self.model attribute
        self.model = model
        model = None

    def transform(self, data):
        """Method for transforming data using trained model.

        Args:
            data(array-like):
                The data that we wish to transform. It must have same
                number of features as self.n_features

        Returns:
            The transformed extracted features fo size n by target_size where
            n = the number of samples.
        """

        data = self.check_data(data)
        encoder = Model(self.model.input, self.model.get_layer('middle_layer').output)
        return encoder.predict(data)
