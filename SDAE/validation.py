
"""Module containing utility functions for checking data."""

import numpy as np
from sklearn.utils.validation import check_array

class InputValidation:
    """ Class containing methods used to check and validate i
    user nput data.

    Attributes:
        None
    """

    def check_data(self, data):
        """Method for enuring correct input dimensions for SDNAE model
        and call SciKit Learn sklearn.utils.check_array function to validate
        array.

        Args:
            data(array-like):
            The data array to check and validate

        Returns:
            data
        """

        if self.n_features != data.shape[1]:
            raise ValueError(
                """Incompatible Feature Dimension.\n
                Please ensure that the input data has the same
                number of features as 'n_features' attribute
                """)
        data = check_array(data, accept_sparse=True, ensure_min_features=2)
        return data

    def check_custom_layers(self):
        """
        Method for checking and validating self.custom_layers attribute

        Args:
            None

        Returns:
            None
        """

        if not isinstance(self.custom_layers, list):
            raise TypeError("""self.cusom_layers is not type List. Please ensure type is list""")

        count = 0
        for element in self.custom_layers:
            count += 1
            if not isinstance(element, int):
                raise TypeError("""self.custom layers elements need to be type Int.""")

        if count != 7:
            raise ValueError("""Custom_layers list is inccorrect size.\n
                             please ensure len(custom_layer) = 7.""")

        if self.custom_layers[0] != self.n_features or self.custom_layers[0] != self.n_features:
            raise ValueError(""" Input and Output layer must equal n_features.""")

        if self.custom_layers[3] != self.target_size:
            raise ValueError("""Middle layer must equal target_size.""")
