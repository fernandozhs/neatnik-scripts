# NEATnik:
import neatnik
import parameters

# Typing:
from typing        import Dict
from neatnik       import Experiment
from neatnik       import Organism
from numpy.typing  import NDArray
from numpy.ma.core import MaskedArray

# Others:
import pickle       as p
import numpy        as np
import scipy.signal as sp
import mpi4py.MPI   as mpi


class DataSelection(Experiment):
    """ Drives the evolution of a data-selecting artificial neural network. """

    global rank
    global size
    global stop

    def __init__(self) -> None:
        """ Initializes this DataSelection Experiment. """

        super().__init__()

        self.tod = p.load(open('tod.p', 'rb'))
        self.stimuli = self.prepare(self.tod)[rank::size]

        self.baseline = 0.001
        self.filter = sp.butter(10, 45, 'highpass', fs=1./np.diff(self.tod['time']).mean(), output='sos')

        self.vertexes = [
            (0,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0, 13),
            (1,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0, 12),
            (2,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0, 11),
            (3,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0, 10),
            (4,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  9),
            (5,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  8),
            (6,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  7),
            (7,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  6),
            (8,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  5),
            (9,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  4),
            (10, None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  3),
            (11, None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  2),
            (12, None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  1),
            (13, None, neatnik.ENABLED,  neatnik.BIAS,   neatnik.UNITY,     0,  0),
            (14, None, neatnik.ENABLED,  neatnik.OUTPUT, neatnik.HEAVISIDE, 1,  6),
            ]
        self.edges = [
            (None, None, neatnik.ENABLED, neatnik.BIASING, 13,  14, None),
            ]

    def prepare(self, tod : Dict[str, NDArray]) -> NDArray:
        """ Puts time-ordered data into a format which can be passed to Organisms as stimuli. """

        amplitudes = np.lib.stride_tricks.sliding_window_view(tod['amplitudes'], window_shape=10, axis=1)
        pointing = np.lib.stride_tricks.sliding_window_view(tod['pointing'], window_shape=10, axis=0)[:,-1]
        time = np.lib.stride_tricks.sliding_window_view(tod['time'], window_shape=10, axis=0)[:,-1]

        pointing = np.repeat(pointing.reshape(-1,1)[np.newaxis,:], repeats=10, axis=0)
        time = np.repeat(time.reshape(-1,1)[np.newaxis,:], repeats=10, axis=0)
        tags = np.repeat(tod['tags'][:,np.newaxis,np.newaxis], repeats=amplitudes.shape[1], axis=1)

        return np.concatenate((amplitudes, pointing, time, tags), axis=2)

    def mask(self, reactions: NDArray) -> NDArray:
        """ Generates a mask from an Organism's reactions. """

        reactions = np.repeat(reactions, 10, axis=2)
        padding = np.zeros((reactions.shape[0], reactions.shape[1], self.tod['amplitudes'].shape[1] - reactions.shape[2]))
        reactions = np.concatenate((reactions, padding), axis=2)

        i, j, k = np.ogrid[0:reactions.shape[0],0:reactions.shape[1],0:reactions.shape[2]]
        l = k - np.arange(0, reactions.shape[1])[:,np.newaxis]

        return reactions[i,j,l].sum(axis=1) > 5

    def fitness(self, organism: Organism) -> float:
        """ Scores the fitness of the input Organism. """

        if rank == 0:
            comm.bcast(stop, root=0)
            comm.bcast(organism, root=0)

        mask = self.mask(organism.react())
        samples = (~mask).sum()
        power = np.sum(np.ma.masked_array(sp.sosfilt(self.filter, self.tod['amplitudes'][rank::size])**2, mask))

        samples = comm.reduce(samples, op=mpi.SUM, root=0)
        power = comm.reduce(power, op=mpi.SUM, root=0)

        if rank == 0:
            return (samples/self.tod['amplitudes'].size)**2 * np.exp(1. - (power/samples)/self.baseline)

        else:
            return 0.

    def display(self) -> None:
        """ Displays information about this Experiment on the screen. """

        print("Max. Fitness:", "%.4f"%neatnik.Parameters.fitness_threshold, end="\r", flush=True)

        return


comm = mpi.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

experiment = DataSelection()
organism = None
stop = False

if rank == 0:
    experiment.build()
    experiment.run()

    input("\nNEATnik has finished.")

    p.dump(experiment.outcome[-1], open('organism.p', 'wb'))

    stop = True
    comm.bcast(stop, root=0)

else:
    while not(comm.bcast(stop, root=0)):
        organism = comm.bcast(organism, root=0)
        experiment.fitness(organism)

mpi.Finalize()
