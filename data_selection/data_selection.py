import neatnik
import pickle
import numpy as np


# Defines the `DataSelection` `neatnik.Experiment`.
class DataSelection(neatnik.Experiment):
    """ Drives the evolution of a data-selecting artificial neural network. """

    # Loads this `neatnik.Experiment`'s `data` and produces its associated `stimuli`.
    data = np.load('data.npy')
    stimuli = np.insert(data, 0, 1, axis=1)

    # Constructor:
    def __init__(self) -> None:
        """ Initializes this `DataSelection` `neatnik.Experiment`. """

        # Initializes the base `neatnik.Experiment`.
        neatnik.Experiment.__init__(self)

        # Sets the `DataSelection` `neatnik.Parameters`.
        self.parameters.generational_cycles = 200
        self.parameters.population_size = 100
        self.parameters.mutation_attempts = 10
        self.parameters.spawning_attempts = 10
        self.parameters.weight_bound = 2.0
        self.parameters.perturbation_power = 2.0
        self.parameters.initial_activation = neatnik.RELU
        self.parameters.rejection_fraction = 0.3
        self.parameters.stagnation_threshold = 15
        self.parameters.compatibility_threshold = 3.0
        self.parameters.compatibility_weights = [0.5, 1., 1.]
        self.parameters.enabling_link = [0.5, 0.5]
        self.parameters.altering_links = [0.5, 0.5]
        self.parameters.altering_weight = [0.2, 0.7, 0.1]
        self.parameters.adding_link = [0.5, 0.25, 0., 0.25, 0.]
        self.parameters.enabling_node = [0.0, 0.5, 0.5]
        self.parameters.altering_nodes = [1., 0.]
        self.parameters.altering_activation = [1., 0., 0., 0.]
        self.parameters.adding_node = [0.5, 0.5, 0.]
        self.parameters.assimilating_links = [0., 1.]
        self.parameters.assimilating_nodes = [1., 0.]
        self.parameters.assimilating_weight = [0.5, 0.5]
        self.parameters.assimilating_activation = [1., 0.]
        self.parameters.spawning_organism = [0.4, 0.6]

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

    # Map-maker:
    def map(self, data: np.ndarray) -> np.ndarray:
        """ Produces the binned map associated with the input `data`. """

        # Returns the map associated with the input `data`.
        return np.mean(data, axis=0)

    # Noise estimate:
    def noise(self, data: np.ndarray, number_splits: int) -> np.ndarray:
        """ Estimates the noise in the map associated with the input `data`. """

        # Produces sample `maps` from the input `data`.
        maps = np.array([self.map(sample) for sample in np.array_split(data, number_splits)])

        # Returns the pixel-wise estimated map noise.
        return np.std(maps, axis=0)

    # Performance:
    def performance(self, organism: neatnik.Organism) -> float:
        """ Scores the performance of the input `organism`. """

        # The number of samples in which this `neatnik.Experiment`'s `data` will be split for map noise estimation.
        number_splits = 20

        # Extracts the input `organism`'s reactions to this `neatnik.Experiment`'s `stimuli`.
        reactions = organism.react(self.stimuli)

        # Converts the above `reactions` into a data-selecting filter.
        filter = np.array(reactions, dtype=np.bool).flatten()

        # Assigns a score to the input `organism`.
        if filter.sum() < 2*number_splits:
            # Too few `data` realizations selected by the input `organism`.
            score = 0.

        else:
            # Estimates the pixel-wise `map_noise` associated with the `organism`-selected `data` realizations.
            map_noise = self.noise(self.data[filter], number_splits)

            # Computes the input `organism`'s score.
            score = filter.sum() * np.exp(-map_noise).sum()

        # Returns the `organism`'s score.
        return score


# Creates and populates a `DataSelection` `neatnik.Experiment`.
experiment = DataSelection()
experiment.populate()

# Runs the `DataSelection` `neatnik.Experiment`.
experiment.run()

# Extracts the best performing `neatnik.Organism`.
pickle.dump(experiment.graph(), open('organism.p', 'wb'))
