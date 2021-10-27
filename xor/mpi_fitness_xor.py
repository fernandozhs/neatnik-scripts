# NEATnik
import neatnik
from . import parameters

# Others
import pickle
import numpy as np
from mpi4py import MPI


# Initializes MPI.
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Defines the XOR `neatnik.Experiment`.
class XOR(neatnik.Experiment):
    """ Drives the evolution of an 'exclusive or' operator. """

    # Makes XOR MPI-aware.
    global rank
    global size

    # Constructor:
    def __init__(self) -> None:
        """ Initializes this XOR `neatnik.Experiment`. """

        # Initializes the base `neatnik.Experiment`.
        neatnik.Experiment.__init__(self)

        # Sets the base network graph associated with the first generation of XOR `neatnik.Organism`s.
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

    # Fitness metric:
    def fitness(self, organism: neatnik.Organism) -> float:
        """ Scores the fitness of the input `neatnik.Organism`. """

        # Master process:
        if rank == 0:
            # Broadcasts the input organism to the worker processes.
            comm.bcast(kill, root=0)
            comm.bcast(organism, root=0)

        # The input stimuli and expected response of an 'exclusive or' operator.
        stimuli = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]])[rank::size]
        response = np.array([[0], [1], [1], [0]])[rank::size]

        # Extracts the input organism's reactions to the above stimuli.
        reactions = organism.react(stimuli)

        # Computes the organism's partial score by comparing its reactions to the expected response.
        score = np.shape(stimuli)[0] - np.abs(reactions - response).flatten().sum()

        # Sums the input organism's scores obtained by each process.
        score = comm.reduce(score, op=MPI.SUM, root=0)

        # Returns the organism's score.
        return score

    # Monitoring:
    def display(self) -> None:
        """ Displays information about this `neatnik.Experiment` on the screen. """

        # Shows the maximum fitness attained.
        print("Max. Fitness:", "%.2f"%self.parameters.fitness_threshold, end="\r", flush=True)

        return


# Initializes a all relevant objects.
experiment = XOR()
organism = None
kill = False

# Master process:
if rank == 0:
    # Sets up and runs the XOR `neatnik.Experiment`.
    experiment.build()
    experiment.run()

    # Extracts the best performing `neatnik.Organism`.
    pickle.dump(experiment.outcome[-1], open('organism.p', 'wb'))

    # Kills all worker processes.
    kill = True
    comm.bcast(kill, root=0)

# Worker processes:
else:
    # Work until told not to.
    while not(comm.bcast(kill, root=0)):
        # Scores the broadcasted organism.
        organism = comm.bcast(organism, root=0)
        experiment.fitness(organism)

# Finalizes MPI.
MPI.Finalize()
