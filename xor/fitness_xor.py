# NEATnik
import neatnik
from . import parameters

# Others
import pickle
import numpy as np


# Defines the XOR `neatnik.Experiment`.
class XOR(neatnik.Experiment):
    """ Drives the evolution of an 'exclusive or' operator. """

    # Constructor:
    def __init__(self) -> None:
        """ Initializes this XOR `neatnik.Experiment`. """

        # Initializes the base `neatnik.Experiment`.
        neatnik.Experiment.__init__(self)

        # Sets the base network graph associated with the first generation of XOR `neatnik.Organism`s.
        self.vertexes = [
            (0, None, neatnik.ENABLED, neatnik.BIAS,   neatnik.IDENTITY, 0, 2),
            (1, None, neatnik.ENABLED, neatnik.INPUT,  neatnik.IDENTITY, 0, 1),
            (2, None, neatnik.ENABLED, neatnik.INPUT,  neatnik.IDENTITY, 0, 0),
            (3, None, neatnik.ENABLED, neatnik.OUTPUT, neatnik.LOGISTIC, 1, 1),
            ]
        self.edges = [
            (None, None, neatnik.ENABLED, neatnik.BIASING, 0, 3, None),
            (None, None, neatnik.ENABLED, neatnik.FORWARD, 1, 3, None),
            (None, None, neatnik.ENABLED, neatnik.FORWARD, 2, 3, None),
            ]

    # Fitness metric:
    def fitness(self, organism: neatnik.Organism) -> float:
        """ Scores the fitness of the input `neatnik.Organism`. """

        # The input stimuli and expected responses of an 'exclusive or' operator.
        stimuli = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
        responses = np.array([[0], [1], [1], [0]])

        # Extracts the input organism's reactions to the above stimuli.
        reactions = organism.react(stimuli)

        # Computes the organism's score by comparing its reactions to the expected responses.
        score = 4 - np.abs(reactions - responses).flatten().sum()

        # Returns the input organism's score.
        return score

    # Monitoring:
    def display(self) -> None:
        """ Displays information about this `neatnik.Experiment` on the screen. """

        # Shows the maximum fitness attained.
        print("Max. Fitness:", "%.2f"%self.parameters.fitness_threshold, end="\r", flush=True)

        return


# Sets up and runs the XOR `neatnik.Experiment`.
experiment = XOR()
experiment.build()
experiment.run()

# Extracts the best performing `neatnik.Organism`.
pickle.dump(experiment.outcome[-1], open('organism.p', 'wb'))
