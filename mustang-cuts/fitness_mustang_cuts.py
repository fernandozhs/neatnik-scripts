# NEATnik:
import neatnik
import parameters

# Minkasi:
import minkasi

# Typing:
from typing        import Dict
from neatnik       import Experiment
from neatnik       import Organism
from numpy.typing  import NDArray
from numpy.ma.core import MaskedArray

# Others:
import numpy  as np
import pickle as p


def get_data(file : str) -> Dict[str, Union[float, NDArray]]:
    """ Loads and readies a MUSTANG data file. """

    # Loads the input data file.
    data = minkasi.read_tod_from_fits(file)

    # Downsamples and truncates the loaded data.
    minkasi.truncate_tod(data)
    minkasi.downsample_tod(data)
    minkasi.truncate_tod(data)
    data['dat_calib'] = minkasi.fit_cm_plus_poly(data['dat_calib'])

    # Returns the readied data.
    return data


# Defines the MustangCuts Experiment.
class MustangCuts(Experiment):
    """ Drives the evolution of a data-cutting artificial neural network. """

    # Loads this Experiment's data.
    data = get_data('/Users/Fernando/Documents/Data/MUSTANG/moo0135/Signal_TOD-AGBT18B_215_01-s8.fits')

    # The stimuli to which all Organisms will react.
    stimuli = np.insert(data.reshape(-1, 10), 0, 1, axis=1)

    # The desired maximum noise level resulting from the data selection.
    noise_target = 0.1

    def __init__(self) -> None:
        """ Initializes this MustandCuts Experiment. """

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

    def power(self, mask: MaskedArray) -> MaskedArray:
        """ Estimates the total power in the high-frequency components of the Experiment's data. """

        # ...

        return #

    def fitness(self, organism: Organism) -> float:
        """ Scores the fitness of the input Organism. """

        # ...

        return #

    def display(self) -> None:
        """ Displays information about this Experiment on the screen. """

        # Shows the maximum fitness attained.
        print("Max. Fitness:", "%.2f"%self.parameters.fitness_threshold, end="\r", flush=True)

        return


# Sets up and and runs the MustangCuts Experiment.
experiment = MustangCuts()
experiment.build()
experiment.run()

# Hangs until the return key is pressed.
input("\nNEATnik has finished.")

# Extracts the best performing Organism.
p.dump(experiment.outcome[-1], open('organism.p', 'wb'))
