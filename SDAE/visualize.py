
"""Module for plotting loss as a function of epochs after training"""

import matplotlib.pyplot as plt
import numpy as np

class Plot:
    """Class for visualising model training

    Attributes:
        None
    """

    def plot_loss(self):
        """Method for plotting training loss as a function
        of epochs

        Args:
            None

        Returns:
            None
        """
        
        epoch_array = np.array(self.model.history.epoch)+1
        loss_array = np.array(self.model.history.history['loss'])
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epochs', fontsize=15)
        ax1.set_ylabel('Loss', color=color, fontsize=15)
        ax1.plot(epoch_array, loss_array, color=color,lw=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)
        plt.title(self.loss, fontsize=15)
        plt.show()
