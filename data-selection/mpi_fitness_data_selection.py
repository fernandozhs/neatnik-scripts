# NEATnik:
import neatnik
import parameters

# Typing:
from neatnik       import Experiment
from neatnik       import Organism
from numpy.typing  import NDArray
from numpy.ma.core import MaskedArray

# Others:
from   mpi4py import MPI
import numpy  as np
import pickle as p


# Defines the DataSelection Experiment.
class DataSelection(Experiment):
    """ Drives the evolution of a data-selecting artificial neural network. """

    # Makes DataSelection MPI-aware.
    global rank
    global size
    global stop

    # Loads this Experiment's data.
    data = np.load('data.npy')

    # The stimuli to which all Organisms will react.
    stimuli = data.reshape(-1, 10)

    # The desired maximum noise level resulting from the data selection.
    noise_target = 0.1

    def __init__(self) -> None:
        """ Initializes this DataSelection Experiment. """

        # Initializes the base Experiment.
        super().__init__()

        # Sets the base network graph associated with the first generation of Organisms.
        self.vertexes = [
            (0,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0, 10),
            (1,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  9),
            (2,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  8),
            (3,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  7),
            (4,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  6),
            (5,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  5),
            (6,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  4),
            (7,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  3),
            (8,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  2),
            (9,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  1),
            (10, None, neatnik.ENABLED,  neatnik.BIAS,   neatnik.UNITY,     0,  0),
            (11, None, neatnik.ENABLED,  neatnik.OUTPUT, neatnik.HEAVISIDE, 1,  5),
            ]
        self.edges = [
            (None, None, neatnik.ENABLED, neatnik.BIASING, 10,  11, None),
            ]

    def mask(self, data: NDArray, reactions: NDArray) -> MaskedArray:
        """ Masks the input data according to the given Organism's reactions. """

        # Creates a mask with shape matching that of the data.
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

        # Master process:
        if rank == 0:
            # Broadcasts the input Organism to the worker processes.
            comm.bcast(stop, root=0)
            comm.bcast(organism, root=0)

        # Extracts the Organism's reactions to the Experiment's stimuli.
        reactions = np.array(organism.react(self.stimuli[rank::size]), dtype=bool)

        # Extracts the data associated with this process.
        data = self.stimuli[rank::size].reshape(-1, self.data.shape[1])

        # Masks and splits the data allocated to this process, producing an average per data split.
        masked_data = self.mask(data, reactions)
        averages = self.average(masked_data, 3)

        # Gathers all reactions and averages.
        reactions = comm.gather(reactions, root=0)
        averages = comm.gather(averages, root=0)

        # Master process:
        if rank == 0:
            # Recasts and reshapes the gathered reactions and noise estimates.
            reactions = np.array(reactions).reshape(self.stimuli.shape[0], -1)
            averages = np.array(averages).reshape(-1, self.data.shape[1])

            # Gets the point-wise noise estimate associated with the gathered averages.
            noise_estimate = self.noise(averages)

            # Scores the Organism's behavior.
            score = (~reactions).sum() * np.exp(-noise_estimate.mean()/self.noise_target)

            # Returns the Organism's score.
            return score

        # Worker process:
        else:
            # No extra work needed.
            return 0.

    def display(self) -> None:
        """ Displays information about this Experiment on the screen. """

        # Shows the maximum fitness attained.
        print("Max. Fitness:", "%.2f"%self.parameters.fitness_threshold, end="\r", flush=True)

        return


# Initializes MPI.
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Initializes a all relevant objects.
experiment = DataSelection()
organism = None
stop = False

# Master process:
if rank == 0:
    # Sets up and runs the XOR Experiment.
    experiment.build()
    experiment.run()

    # Hangs until the return key is pressed.
    input("\nNEATnik has finished.")

    # Extracts the best performing Organism.
    p.dump(experiment.outcome[-1], open('organism.p', 'wb'))

    # Kills all worker processes.
    stop = True
    comm.bcast(stop, root=0)

# Worker processes:
else:
    # Work until told not to.
    while not(comm.bcast(stop, root=0)):
        # Scores the broadcasted organism.
        organism = comm.bcast(organism, root=0)
        experiment.fitness(organism)

# Finalizes MPI.
MPI.Finalize()
