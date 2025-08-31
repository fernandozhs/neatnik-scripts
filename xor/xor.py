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

        nodes = [
            (0, neatnik.ENABLED, neatnik.INPUT,  neatnik.IDENTITY, 0, 2),
            (1, neatnik.ENABLED, neatnik.INPUT,  neatnik.IDENTITY, 0, 1),
            (2, neatnik.ENABLED, neatnik.BIAS,   neatnik.UNITY,    0, 0),
            (3, neatnik.ENABLED, neatnik.OUTPUT, neatnik.LOGISTIC, 1, 1),
            ]
        links = [
            (None, neatnik.ENABLED, neatnik.BIASING, None, 2, 3),
            ]
        population = (nodes,links)
        self.set(population)

        stimuli = np.array([[[0, 0], [1, 0], [0, 1], [1, 1]]])
        self.set(stimuli)

        self.responses = np.array([[[0], [1], [1], [0]]])

    def fitness(self, organism: Organism) -> float:
        """ Scores the fitness of the input Organism. """

        reactions = organism.react()

        score = 4 - np.abs(reactions - self.responses).sum()

        return score

    def display(self) -> None:
        """ Displays information about the Experiment on the screen. """

        max_score = experiment.genus.species[neatnik.DOMINANT][0].organisms[neatnik.DOMINANT][0].score

        print(f"Max. Fitness: {max_score:.2f}", end="\r", flush=True)

        return


experiment = XOR()
experiment.initialize()
experiment.run()

if experiment.MPI_rank == 0:
    organism = experiment.genus.species[neatnik.DOMINANT][0].organisms[neatnik.DOMINANT][0];
    p.dump(organism.data(), open('./xor.p', 'wb'))

experiment.finalize()