# NEATnik:
import neatnik
import parameters

# Typing:
from neatnik import Experiment
from neatnik import Organism

# Others:
import pickle       as p
import numpy        as np
import scipy.signal as sp


class DataSelection(Experiment):
    """ Drives the evolution of a data-selecting artificial neural network. """

    def __init__(self) -> None:
        """ Initializes this DataSelection Experiment. """

        super().__init__()

        self.prepare('./tod.p')

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

        self.target = 0.001

    def prepare(self, file : str) -> None:
        """ Puts the raw time-ordered data into a format which can be used by the algorithm. """

        tod = p.load(open(file, 'rb'))

        # Puts the raw time-ordered data into a more appropriate format.
        amplitudes = np.lib.stride_tricks.sliding_window_view(tod['amplitudes'], window_shape=10, axis=1)

        pointing = np.lib.stride_tricks.sliding_window_view(tod['pointing'], window_shape=10, axis=0)[:,-1]
        pointing = np.repeat(pointing.reshape(-1,1)[np.newaxis,:], repeats=10, axis=0)

        time = np.lib.stride_tricks.sliding_window_view(tod['time'], window_shape=10, axis=0)[:,-1]
        time = np.repeat(time.reshape(-1,1)[np.newaxis,:], repeats=10, axis=0)

        tags = np.repeat(tod['tags'][:,np.newaxis,np.newaxis], repeats=amplitudes.shape[1], axis=1)

        # Collects the above into a single object which can be passed to Organisms as stimuli.
        self.stimuli = np.concatenate((amplitudes, pointing, time, tags), axis=2)

        # Stores the power density of the filtered raw time-ordered amplitudes.
        filter = sp.butter(10, 45, 'highpass', fs=1./np.diff(tod['time']).mean(), output='sos')
        self.data = sp.sosfilt(filter, tod['amplitudes'])**2

        return

    def fitness(self, organism: Organism) -> float:
        """ Scores the fitness of the input Organism. """

        # Extracts the Organism's reactions.
        reactions = organism.react()

        # Initializes the total number of samples retained and their associated power.
        power = 0
        samples = self.data.size - np.sum(reactions)

        # Translates the reactions into cuts.
        reactions = [np.ma.clump_unmasked(reaction) for reaction in np.ma.masked_array(reactions, reactions)]

        # Computes the total amount of power in the kept samples.
        for amplitudes, cuts in zip(self.data, reactions):
            for cut in cuts:
                power += amplitudes[cut].sum()

        return (samples/self.data.size)**2 * np.exp(1. - (power/samples)/self.target)

    def display(self) -> None:
        """ Displays information about this Experiment on the screen. """

        max_score = experiment.genus.species[neatnik.DOMINANT][0].organisms[neatnik.DOMINANT][0].score

        print("Max. Fitness:", "%.6f"%max_score, end="\r", flush=True)

        return


experiment = DataSelection()
experiment.run()

if experiment.MPI_rank == 0:

    organism = experiment.genus.species[neatnik.DOMINANT][0].organisms[neatnik.DOMINANT][0];
    p.dump(organism.graph(), open('/scratch/r/rbond/fzhs/data_selection_organism.p', 'wb'))
