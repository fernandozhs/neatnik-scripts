# NEATnik:
import neatnik
import parameters

# Typing:
from neatnik import Experiment
from neatnik import Organism

# Others:
import numpy  as np
import pickle as p


class XOR(Experiment):
    """ Drives the evolution of an 'exclusive or' operator. """

    def __init__(self) -> None:
        """ Initializes the XOR Experiment. """

        super().__init__()

        self.vertexes = [
            (0, None, neatnik.ENABLED, neatnik.INPUT,  neatnik.IDENTITY, 0, 2),
            (1, None, neatnik.ENABLED, neatnik.INPUT,  neatnik.IDENTITY, 0, 1),
            (2, None, neatnik.ENABLED, neatnik.BIAS,   neatnik.UNITY,    0, 0),
            (3, None, neatnik.ENABLED, neatnik.OUTPUT, neatnik.LOGISTIC, 1, 1),
            ]
        self.edges = [
            (None, None, neatnik.ENABLED, neatnik.BIASING, 2, 3, None),
            ]

        self.stimuli = np.array([[[0, 0], [1, 0], [0, 1], [1, 1]]])
        self.responses = np.array([[[0], [1], [1], [0]]])

    def fitness(self, organism: Organism) -> float:
        """ Scores the fitness of the input Organism. """

        reactions = organism.react()

        score = 4 - np.abs(reactions - self.responses).flatten().sum()

        return score

    def display(self) -> None:
        """ Displays information about the Experiment on the screen. """

        max_score = experiment.genus.species[neatnik.DOMINANT][0].organisms[neatnik.DOMINANT][0].score

        print("Max. Fitness:", "%.2f"%max_score, end="\r", flush=True)

        return


experiment = XOR()
experiment.run()

if experiment.MPI_rank == 0:

    organism = experiment.genus.species[neatnik.DOMINANT][0].organisms[neatnik.DOMINANT][0];
    p.dump(organism.graph(), open('/scratch/r/rbond/fzhs/xor_organism.p', 'wb'))
