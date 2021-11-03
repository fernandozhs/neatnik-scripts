# NEATnik:
import neatnik
import parameters

# Typing:
from neatnik       import Experiment
from neatnik       import Organism
from numpy.typing  import NDArray
from numpy.ma.core import MaskedArray

# Others:
import numpy  as np
import pickle as p


# Defines the DataSelection Experiment.
class DataSelection(Experiment):
    """ Drives the evolution of a data-selecting artificial neural network. """

    # Loads this Experiment's data.
    data = np.load('data.npy')

    # The stimuli to which all Organisms will react.
    stimuli = np.insert(data.reshape(-1, 10), 0, 1, axis=1)

    # The desired maximum noise level resulting from the data selection.
    noise_target = 0.1

    def __init__(self) -> None:
        """ Initializes this DataSelection Experiment. """

        # Initializes the base Experiment.
        super().__init__()

        # Sets the base network graph associated with the first generation of Organisms.
        self.vertexes = [
            (0,  None, neatnik.ENABLED,  neatnik.BIAS,   neatnik.IDENTITY,  0, 10),
            (1,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  9),
            (2,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  8),
            (3,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  7),
            (4,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  6),
            (5,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  5),
            (6,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  4),
            (7,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  3),
            (8,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  2),
            (9,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  1),
            (10, None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  0),
            (11, None, neatnik.ENABLED,  neatnik.OUTPUT, neatnik.HEAVISIDE, 1,  5),
            ]
        self.edges = [
            (None, None, neatnik.ENABLED, neatnik.BIASING, 0,  11, None),
            ]

    def mask(self, data: NDArray, reactions: NDArray) -> MaskedArray:
        """ Masks the input data according to the given Organism's reactions. """

        # Creates a mask with shape matching that of the input data.
        mask = np.tile(reactions, 10).reshape(data.shape)

        # Returns the input data combined with its associated mask.
        return np.ma.masked_array(data, mask)

    def average(self, data: MaskedArray, number_splits: int) -> MaskedArray:
        """ Splits the input data and extracts the point-wise average from each split. """

        # Returns the point-wise average for each data split.
        return np.array([split.mean(axis=0) for split in np.array_split(data, number_splits)])

    def noise(self, averages: MaskedArray) -> MaskedArray:
        """ Estimates the point-wise signal noise by computing the standard deviation of several split averages. """

        # Returns the point-wise estimated signal noise.
        return averages.std(axis=0)

    def fitness(self, organism: Organism) -> float:
        """ Scores the fitness of the input Organism. """

        # Extracts the Organism's reactions to the Experiment's stimuli.
        reactions = np.array(organism.react(self.stimuli), dtype=bool)

        # Masks this Experiment's data according to the above reactions.
        masked_data = self.mask(self.data, reactions)

        # Splits this Experiment's and extracts an average signal per data split.
        averages = self.average(masked_data, 12)

        # Estimates the point-wise signal noise from all averages.
        noise_estimate = self.noise(averages)

        # Scores the Organism's fitness.
        score = (~reactions).sum() * np.exp(-noise_estimate.mean()/self.noise_target)

        # Returns the Organism's score.
        return score

    def display(self) -> None:
        """ Displays information about this Experiment on the screen. """

        # Shows the maximum fitness attained.
        print("Max. Fitness:", "%.2f"%self.parameters.fitness_threshold, end="\r", flush=True)

        return


# Sets up and and runs the DataSelection Experiment.
experiment = DataSelection()
experiment.build()
experiment.run()

# Hangs until the return key is pressed.
input("\nNEATnik has finished.")

# Extracts the best performing Organism.
p.dump(experiment.outcome[-1], open('organism.p', 'wb'))
