# NEATnik:
import neatnik
import parameters

# Typing:
from neatnik import Experiment
from neatnik import Organism

# Others:
from   mpi4py import MPI
import numpy  as np
import pickle as p


# Defines the XOR Experiment.
class XOR(Experiment):
    """ Drives the evolution of an 'exclusive or' operator. """

    # Makes XOR MPI-aware.
    global rank
    global size
    global stop

    # The stimuli and expected responses of an Organism behaving as an 'exclusive or' operator.
    stimuli = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
    response = np.array([[0], [1], [1], [0]])

    # Constructor:
    def __init__(self) -> None:
        """ Initializes this XOR Experiment. """

        # Initializes the base Experiment.
        super().__init__()

        # Sets the base network graph associated with the first generation of Organisms.
        self.vertexes = [
            (0,  None, neatnik.ENABLED, neatnik.BIAS,   neatnik.IDENTITY, 0, 2),
            (1,  None, neatnik.ENABLED, neatnik.INPUT,  neatnik.IDENTITY, 0, 1),
            (2,  None, neatnik.ENABLED, neatnik.INPUT,  neatnik.IDENTITY, 0, 0),
            (3,  None, neatnik.ENABLED, neatnik.OUTPUT, neatnik.LOGISTIC, 1, 1),
            ]
        self.edges = [
            (None, None, neatnik.ENABLED, neatnik.BIASING, 0, 3, None),
            (None, None, neatnik.ENABLED, neatnik.FORWARD, 1, 3, None),
            (None, None, neatnik.ENABLED, neatnik.FORWARD, 2, 3, None),
            ]

    def fitness(self, organism: Organism) -> float:
        """ Scores the fitness of the input Organism. """

        # Master process:
        if rank == 0:
            # Broadcasts the current Organism to the worker processes.
            comm.bcast(stop, root=0)
            comm.bcast(organism, root=0)

        # Extracts the Organism's reactions to the Experiment's stimuli.
        reactions = organism.react(self.stimuli[rank::size])

        # Computes the Organism's partial score by comparing its reactions to the expected behavior of an 'exclusive or' operator.
        score = np.shape(self.stimuli[rank::size])[0] - np.abs(reactions - self.response[rank::size]).flatten().sum()

        # Sums the partial scores obtained by each process.
        score = comm.reduce(score, op=MPI.SUM, root=0)

        # Returns the Organism's total score.
        return score

    def display(self) -> None:
        """ Displays information about this Experiment on the screen. """

        # Shows the maximum fitness attained.
        print("Max. Fitness:", "%.2f"%self.parameters.fitness_threshold, end="\r", flush=True)

        return


# Initializes MPI.
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Initializes all relevant objects.
experiment = XOR()
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

    # Stops all worker processes.
    stop = True
    comm.bcast(stop, root=0)

# Worker processes:
else:
    # Works until told not to.
    while not(comm.bcast(stop, root=0)):
        # Scores the broadcasted Organism.
        organism = comm.bcast(organism, root=0)
        experiment.fitness(organism)

# Finalizes MPI.
MPI.Finalize()
