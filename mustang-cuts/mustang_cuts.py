# NEATnik:
import neatnik
import parameters

# Minkasi:
import minkasi

# Typing:
from neatnik import Experiment
from neatnik import Organism

# Others:
import operator
import pickle as p
import numpy  as np


class MustangCuts(neatnik.Experiment):
    """ Drives the evolution of a data-selecting artificial neural network for Mustang. """

    def __init__(self) -> None:
        """ Initializes the MustangCuts Experiment. """

        super().__init__()

        self.prepare('/project/s/sievers/sievers/mustang/data/Zw3146/Signal_TOD-AGBT18A_175_01-s11.fits')
        self.target = 79776.12195529531

        self.vertexes = [
            (0,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0, 15),
            (1,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0, 14),
            (2,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0, 13),
            (3,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0, 12),
            (4,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0, 11),
            (5,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0, 10),
            (6,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  9),
            (7,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  8),
            (8,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  7),
            (9,  None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  6),
            (10, None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  5),
            (11, None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  4),
            (12, None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  3),
            (13, None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  2),
            (14, None, neatnik.DISABLED, neatnik.INPUT,  neatnik.IDENTITY,  0,  1),
            (15, None, neatnik.ENABLED,  neatnik.BIAS,   neatnik.UNITY,     0,  0),
            (16, None, neatnik.ENABLED,  neatnik.OUTPUT, neatnik.HEAVISIDE, 1,  8),
            ]
        self.edges = [
            (None, None, neatnik.ENABLED, neatnik.BIASING, 15,  16, None),
            ]

    def prepare(self, file : str) -> None:
        """ Puts time-ordered data into a format which can be understood by the algorithm. """ 

        # Loads the MUSTANG time-ordered data from the input FITS file.
        tod = minkasi.read_tod_from_fits(path)
        minkasi.truncate_tod(tod)
        minkasi.downsample_tod(tod)
        minkasi.truncate_tod(tod)
        tod['dat_calib'] = minkasi.fit_cm_plus_poly(tod['dat_calib'])
        tod = minkasi.Tod(tod)

        # Restrict the detector timestreams to the following:
        detectors = [1,2,3,4,77,78,130]

        # Puts the raw time-ordered data into a more appropriate format.
        dat = np.lib.stride_tricks.sliding_window_view(tod.info['dat_calib'][detectors], window_shape=10, axis=1)
        dat = dat/np.max(tod.info['dat_calib'])

        dx = np.lib.stride_tricks.sliding_window_view(tod.info['dx'][detectors], window_shape=10, axis=1)[:,:,4:5]
        dx = dx/np.max(tod.info['dx'])

        dy = np.lib.stride_tricks.sliding_window_view(tod.info['dy'][detectors], window_shape=10, axis=1)[:,:,4:5]
        dy = dx/np.max(tod.info['dy'])

        elev = np.lib.stride_tricks.sliding_window_view(tod.info['elev'][detectors], window_shape=10, axis=1)[:,:,4:5]
        elev = dx/np.max(tod.info['elev'])

        time = np.linspace(0, 1, tod.info['dat_calib'].shape[1])
        time = np.lib.stride_tricks.sliding_window_view(time, window_shape=10, axis=0)[:,4]
        time = np.repeat(time.reshape(-1,1)[np.newaxis,:], repeats=dat.shape[0], axis=0)

        pixid = np.repeat(tod.info['pixid'][:,np.newaxis,np.newaxis], repeats=dat.shape[1], axis=1)[detectors]
        pixid = pixid/np.max(tod.info['pixid'])

        # Collects the above into a single object which can be passed to Organisms as stimuli.
        self.stimuli = np.concatenate((dat, dx, dy, elev, time, pixid), axis=2)

        # Stores the power density of the filtered raw time-ordered amplitudes.
        tod.set_noise(minkasi.NoiseSmoothedSVD)
        self.data = (tod.apply_noise(tod.info['dat_calib'])**2)[detectors].flatten()

        return

    def fitness(self, organism : Organism) -> float:
        """ Scores the fitness of the input Organism. """

        # Extracts the Organism's reactions.
        reactions = organism.react()

        # Initializes the total number of samples retained and their associated power.
        power = 0
        samples = self.data.size - np.sum(reactions)

        # Translates the reactions into cuts.
        uncut = np.ma.clump_unmasked(np.ma.masked_array(reactions, reactions))

        # Computes the amount of uncut power.
        if len(uncut) > 0: power = sum(map(np.sum, operator.itemgetter(*uncut)(self.data)))

        return (samples/self.data.size)**2 * np.exp(1. - (power/samples)/self.target)

    def display(self) -> None:
        """ Displays information about the Experiment on the screen. """

        # The maximum fitness score attained.
        max_score = experiment.genus.species[neatnik.DOMINANT][0].organisms[neatnik.DOMINANT][0].score

        print("Max. Fitness:", "%.6f"%max_score, end="\r", flush=True)

        return


experiment = MustangCuts()
experiment.run()

if experiment.MPI_rank == 0:
    organism = experiment.genus.species[neatnik.DOMINANT][0].organisms[neatnik.DOMINANT][0];
    p.dump(organism.graph(), open('/scratch/r/rbond/fzhs/mustang_cuts_organism.p', 'wb'))
