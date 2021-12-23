# NEATnik:
import neatnik
import parameters

# Typing:
from neatnik import Experiment
from neatnik import Organism

# Others:
import numpy  as np
import pickle as p


# Defines the XOR Experiment.
class XOR(Experiment):
    """ Drives the evolution of an 'exclusive or' operator. """

    # The stimuli and expected responses of an Organism behaving as an 'exclusive or' operator.
    stimuli = np.array([[[0, 0], [1, 0], [0, 1], [1, 1]]])
    responses = np.array([[[0], [1], [1], [0]]])

    def __init__(self) -> None:
        """ Initializes this XOR Experiment. """

        # Initializes the base Experiment.
        super().__init__()

        # Sets the base network graph associated with the first generation of Organisms.
        self.vertexes = [
            (0, None, neatnik.ENABLED, neatnik.INPUT,  neatnik.IDENTITY, 0, 2),
            (1, None, neatnik.ENABLED, neatnik.INPUT,  neatnik.IDENTITY, 0, 1),
            (2, None, neatnik.ENABLED, neatnik.BIAS,   neatnik.UNITY,    0, 0),
            (3, None, neatnik.ENABLED, neatnik.OUTPUT, neatnik.LOGISTIC, 1, 1),
            ]
        self.edges = [
            (None, None, neatnik.ENABLED, neatnik.BIASING, 2, 3, None),
            ]

    def fitness(self, organism: Organism) -> float:
        """ Scores the fitness of the input Organism. """

        # Extracts the Organism's reactions to the Experiment's stimuli.
        reactions = organism.react(self.stimuli)

        # Computes the Organism's score by comparing its reactions to the expected responses.
        score = 4 - np.abs(reactions - self.responses).flatten().sum()

        # Returns the Organism's score.
        return score

    def display(self) -> None:
        """ Displays information about this Experiment on the screen. """

        # Shows the maximum fitness attained.
        print("Max. Fitness:", "%.2f"%self.parameters.fitness_threshold, end="\r", flush=True)

        return


# Sets up and runs the XOR Experiment.
experiment = XOR()
experiment.build()
experiment.run()

# Hangs until the return key is pressed.
input("\nNEATnik has finished.")

# Extracts the best performing Organism.
p.dump(experiment.outcome[-1], open('organism.p', 'wb'))
