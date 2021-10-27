# NEATnik
import neatnik
from . import parameters

# Others
import pickle
import numpy as np
import numpy.ma as ma
from mpi4py import MPI


# Initializes MPI.
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Defines the DataSelection `neatnik.Experiment`.
class DataSelection(neatnik.Experiment):
    """ Drives the evolution of a data-selecting artificial neural network. """

    # Makes DataSelection MPI-aware.
    global rank
    global size

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

        # Creates a mask with shape matching that of the data.
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

        # Master process:
        if rank == 0:
            # Broadcasts the input organism to the worker processes.
            comm.bcast(kill, root=0)
            comm.bcast(organism, root=0)

        # Extracts the input organism's reactions to the stimuli associated with this process.
        reactions = np.array(organism.react(self.stimuli[rank::size]), dtype=np.bool)

        # Extracts the data associated with this process.
        data = np.delete(self.stimuli[rank::size], 0, 1).reshape(-1, self.data.shape[1])

        # Masks and splits the data allocated to this process, producing a map per data split.
        masked_data = self.mask(data, reactions)
        maps = self.map(masked_data, 3)

        # Gathers all reactions and maps.
        reactions = comm.gather(reactions, root=0)
        maps = comm.gather(maps, root=0)

        # Master process:
        if rank == 0:
            # Recasts and reshapes the gathered reactions and noite estimates.
            reactions = np.array(reactions).reshape(self.stimuli.shape[0], -1)
            maps = np.array(maps).reshape(-1, self.data.shape[1])

            # Gets the pixel-wise noise estimate associated with the gathered maps.
            noise_estimate = self.noise(maps)

            # Scores the input organism's behavior.
            score = (~reactions).sum() * np.exp(-noise_estimate.mean()/self.noise_target)

            # Returns the organism's score.
            return score

        # Worker process:
        else:
            # No extra work needed.
            return 0.

    # Monitoring:
    def display(self) -> None:
        """ Displays information about this `neatnik.Experiment` on the screen. """

        # Shows the maximum fitness attained.
        print("Max. Fitness:", "%.2f"%self.parameters.fitness_threshold, end="\r", flush=True)

        return


# Initializes a all relevant objects.
experiment = DataSelection()
organism = None
kill = False

# Master process:
if rank == 0:
    # Sets up and runs the XOR `neatnik.Experiment`.
    experiment.build()
    experiment.run()

    # Extracts the best performing `neatnik.Organism`.
    pickle.dump(experiment.outcome[-1], open('organism.p', 'wb'))

    # Kills all worker processes.
    kill = True
    comm.bcast(kill, root=0)

# Worker processes:
else:
    # Work until told not to.
    while not(comm.bcast(kill, root=0)):
        # Scores the broadcasted organism.
        organism = comm.bcast(organism, root=0)
        experiment.fitness(organism)

# Finalizes MPI.
MPI.Finalize()
