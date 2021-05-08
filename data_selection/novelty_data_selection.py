import neatnik
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

    # Constructor:
    def __init__(self) -> None:
        """ Initializes this DataSelection `neatnik.Experiment`. """

        # Initializes the base `neatnik.Experiment`.
        neatnik.Experiment.__init__(self)

        # Sets the DataSelection `neatnik.Parameters`.
        self.parameters.evolution_driver = neatnik.NOVELTY
        self.parameters.fitness_threshold = 0.
        self.parameters.novelty_threshold = 1.0
        self.parameters.novelty_neighbors = 5
        self.parameters.novelty_threshold_modifiers = [1.001, 0.987]
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

    # Cuts:
    def mask(self, data: np.ndarray, reactions: np.ndarray) -> ma.core.MaskedArray:
        """ Masks the input data according to the given `neatnik.Organism` reactions. """

        # Creates a mask with shape matching that of the input data.
        mask = np.tile(reactions, 10).reshape(data.shape)

        # Returns the input data combined with its associated mask.
        return ma.masked_array(data, mask)

    # Map-maker:
    def map(self, data: ma.core.MaskedArray) -> ma.core.MaskedArray:
        """ Maps the input data. """

        # Returns the binned map associated with the input data.
        return data.mean(axis=0)

    # Noise estimate:
    def noise(self, data: ma.core.MaskedArray, number_splits: int) -> ma.core.MaskedArray:
        """ Estimates the noise in the map associated with the input data. """

        # Produces a number of split maps from the input data.
        maps = np.array([self.map(sample) for sample in np.array_split(data, number_splits)])

        # Returns the pixel-wise estimated map noise.
        return maps.std(axis=0)

    # Behavior definition:
    def behavior(self, organism: neatnik.Organism) -> np.ndarray:
        """ Extracts the behavior of the input `neatnik.Organism`. """

        # Extracts the input organism's reactions to this `neatnik.Experiment`'s stimuli.
        reactions = np.array(organism.react(self.stimuli), dtype=np.bool)

        # Masks this `neatnik.Experiment`'s `data` according to the input reactions.
        masked_data = self.mask(self.data, reactions)

        # Gets the pixel-wise noise estimate associated with the masked data.
        noise_estimate = self.noise(masked_data, 100)

        # Returns the input organism's behavior.
        return np.array([(~reactions).sum()/len(reactions), noise_estimate.max()])

    # Fitness metric:
    def fitness(self, organism: neatnik.Organism) -> float:
        """ Scores the fitness of the input `neatnik.Organism`. """

        # Assigns a score to the input organism.
        score = organism.behavior[0]*len(organism.behavior[0])*np.exp(-organism.behavior[1])

        # Returns the organism's score.
        return score

    # Monitoring:
    def display(self) -> None:
        """ Displays information about this `neatnik.Experiment` on the screen. """

        # Shows the maximum fitness attained.
        print("Max. Fitness:", "%.4f"%self.parameters.fitness_threshold, end="\r", flush=True)

        return


# Sets and and runs the DataSelection `neatnik.Experiment`.
experiment = DataSelection()
experiment.build()
experiment.run()

# Saves all discovered behaviors.
np.save('behaviors', np.array(experiment.behaviors))

# Extracts the best performing `neatnik.Organism`.
pickle.dump(experiment.outcome[-1], open('organism.p', 'wb'))
