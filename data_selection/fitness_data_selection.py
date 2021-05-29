# NEATnik
import neatnik
import parameters

# Others
import pickle
import numpy as np
import numpy.ma as ma


# Defines the DataSelection `neatnik.Experiment`.
class DataSelection(neatnik.Experiment):
    """ Drives the evolution of a data-selecting artificial neural network. """

    # Loads this `neatnik.Experiment`'s data.
    data = np.load('data.npy')

    # Produces the stimuli associated with the loaded data.
    stimuli = np.insert(data.reshape(-1, 10), 0, 1, axis=1)

    # The desired maximum noise level resulting from the data selection.
    noise_target = 0.1

    # Constructor:
    def __init__(self) -> None:
        """ Initializes this DataSelection `neatnik.Experiment`. """

        # Initializes the base `neatnik.Experiment`.
        neatnik.Experiment.__init__(self)

        # Sets the base network graph associated with the first generation of `neatnik.Organism`s.
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

    # Cuts:
    def mask(self, data: np.ndarray, reactions: np.ndarray) -> ma.core.MaskedArray:
        """ Masks the input data according to the given `neatnik.Organism` reactions. """

        # Creates a mask with shape matching that of the input data.
        mask = np.tile(reactions, 10).reshape(data.shape)

        # Returns the input data combined with its associated mask.
        return ma.masked_array(data, mask)

    # Map-maker:
    def map(self, data: ma.core.MaskedArray, number_splits: int) -> ma.core.MaskedArray:
        """ Splits the input data and produces a map from each split. """

        # Produces a map per each input data split.
        maps = np.array([split.mean(axis=0) for split in np.array_split(data, number_splits)])

        # Returns the maps associated with each input data split.
        return maps

    # Noise estimate:
    def noise(self, maps: ma.core.MaskedArray) -> ma.core.MaskedArray:
        """ Estimates the pixel-wise noise in a map by computing the standard deviation of several map realizations. """

        # Returns the pixel-wise estimated map noise.
        return maps.std(axis=0)

    # Fitness metric:
    def fitness(self, organism: neatnik.Organism) -> float:
        """ Scores the fitness of the input `neatnik.Organism`. """

        # Extracts the input organism's reactions to this `neatnik.Experiment`'s stimuli.
        reactions = np.array(organism.react(self.stimuli), dtype=np.bool)

        # Masks this `neatnik.Experiment`'s data according to the above reactions.
        masked_data = self.mask(self.data, reactions)

        # Splits this `neatnik.Experiment`'s data and produces a map per data split.
        maps = self.map(masked_data, 12)

        # Estimates the pixel-wise map noise from all map realizations.
        noise_estimate = self.noise(maps)

        # Scores the input organism's fitness.
        score = (~reactions).sum() * np.exp(-noise_estimate.mean()/self.noise_target)

        # Returns the organism's score.
        return score

    # Monitoring:
    def display(self) -> None:
        """ Displays information about this `neatnik.Experiment` on the screen. """

        # Shows the maximum fitness attained.
        print("Max. Fitness:", "%.2f"%self.parameters.fitness_threshold, end="\r", flush=True)

        return

# Sets and and runs the DataSelection `neatnik.Experiment`.
experiment = DataSelection()
experiment.build()
experiment.run()

# Extracts the best performing `neatnik.Organism`.
pickle.dump(experiment.outcome[-1], open('organism.p', 'wb'))
