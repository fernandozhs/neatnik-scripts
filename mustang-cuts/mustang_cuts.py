# NEATnik:
import neatnik
import parameters

# Minkasi:
import minkasi

# Others:
import pickle       as p
import numpy        as np
import scipy.signal as sp


class MustangCuts(neatnik.Experiment):
    """ Drives the evolution of a data-selecting artificial neural network for Mustang. """

    def __init__(self) -> None:
        """ Initializes the MustangCuts Experiment. """

        super().__init__()

        self.tod = self.read('/Users/Fernando/Documents/Data/MUSTANG/moo0135/Signal_TOD-AGBT18B_215_01-s8.fits')
        self.stimuli = self.prepare(self.tod)

        self.baseline = 0.001
        self.filter = sp.butter(10, 10, 'highpass', fs=1./self.tod.info['dt'], output='sos')

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

    def read(self, path):
        """ Reads Mustang time-ordered data from a FITS file. """

        tod = minkasi.read_tod_from_fits(path)

        minkasi.truncate_tod(tod)
        minkasi.downsample_tod(tod)
        minkasi.truncate_tod(tod)

        tod['dat_calib'] = minkasi.fit_cm_plus_poly(tod['dat_calib'])

        return minkasi.Tod(tod)

    def prepare(self, tod):
        """ Puts time-ordered data into a format which can be passed to Organisms as stimuli. """

        dat = np.lib.stride_tricks.sliding_window_view(self.tod.info['dat_calib'], window_shape=10, axis=1)
        dat = dat/np.max(dat)

        dx = np.lib.stride_tricks.sliding_window_view(self.tod.info['dx'], window_shape=10, axis=1)[:,:,4:5]
        dx = dx/np.max(dx)

        dy = np.lib.stride_tricks.sliding_window_view(self.tod.info['dy'], window_shape=10, axis=1)[:,:,4:5]
        dy = dx/np.max(dy)

        elev = np.lib.stride_tricks.sliding_window_view(self.tod.info['elev'], window_shape=10, axis=1)[:,:,4:5]
        elev = dx/np.max(elev)

        time = np.linspace(0, 1, self.tod.info['dat_calib'].shape[1])
        time = np.lib.stride_tricks.sliding_window_view(time, window_shape=10, axis=0)[:,4]
        time = np.repeat(time.reshape(-1,1)[np.newaxis,:], repeats=dat.shape[0], axis=0)

        pixid = np.repeat(self.tod.info['pixid'][:,np.newaxis,np.newaxis], repeats=dat.shape[1], axis=1)
        pixid = pixid/np.max(pixid)

        self.tod.set_noise(minkasi.NoiseSmoothedSVD)
        self.tod.info['dat_calib'] = self.tod.apply_noise(self.tod.info['dat_calib'])

        return np.concatenate((dat, dx, dy, elev, time, pixid), axis=2)

    def mask(self, reactions):
        """ Generates a mask from an Organism's reactions. """

        reactions = np.repeat(reactions, 10, axis=2)
        padding = np.zeros((reactions.shape[0], reactions.shape[1], self.tod.info['dat_calib'].shape[1] - reactions.shape[2]))
        reactions = np.concatenate((reactions, padding), axis=2)

        i, j, k = np.ogrid[0:reactions.shape[0],0:reactions.shape[1],0:reactions.shape[2]]
        l = k - np.arange(0, reactions.shape[1])[:,np.newaxis]

        return reactions[i,j,l].sum(axis=1) > 5

    def fitness(self, organism):
        """ Scores the fitness of the input Organism. """

        mask = self.mask(organism.react())
        samples = (~mask).sum()
        power = np.sum(np.ma.masked_array(sp.sosfilt(self.filter, self.tod.info['dat_calib'])**2, mask))

        return (samples/self.tod.info['dat_calib'].size)**2 * np.exp(1. - (power/samples)/self.baseline)

    def display(self):
        """ Displays information about this Experiment on the screen. """

        max_score = experiment.genus.species[neatnik.DOMINANT][0].organisms[neatnik.DOMINANT][0].score

        print("Max. Fitness:", "%.6f"%max_score, end="\n", flush=True)

        return


experiment = MustangCuts()
experiment.run()

if experiment.MPI_rank == 0:

    organism = experiment.genus.species[neatnik.DOMINANT][0].organisms[neatnik.DOMINANT][0];
    p.dump(organism.graph(), open('mustang_organism.p', 'wb'))
