# NEATnik:
import neatnik
import parameters

# Typing:
from neatnik import Experiment
from neatnik import Organism

# Others:
import numpy  as np
import pickle as p


class AR1(Experiment):
    """ Drives the evolution of an AR1 process forecaster. """

    def __init__(self) -> None:
        """ Initializes the AR1 Experiment. """

        super().__init__()

        nodes = [
            (0, neatnik.ENABLED, neatnik.INPUT,  neatnik.IDENTITY, 0, 1),
            (1, neatnik.ENABLED, neatnik.BIAS,   neatnik.UNITY,    0, 0),
            (2, neatnik.ENABLED, neatnik.OUTPUT, neatnik.IDENTITY, 1, 1),
            ]
        links = [
            (None, neatnik.ENABLED, neatnik.BIASING, None, 1, 2),
            ]
        population = (nodes,links)
        self.set(population)

        with open('../data.p', 'rb') as file:
            stimuli = p.load(file)
        self.set(stimuli)

        self.scale = np.abs(self.stimuli).sum()

        self.cycle_counter = 1
        self.species_counter = 1

    def fitness(self, organism: Organism) -> float:
        """ Scores the fitness of the input Organism. """

        reactions = organism.react()

        deviation = np.abs(self.stimuli[:,:-1,:] - reactions[:,1:,:]).sum()
        score = np.exp(-deviation/self.scale)

        return score

    def display(self) -> None:
        """ Displays information (from a single MPI rank) about the Experiment on the screen. """

        max_score = experiment.genus.species[neatnik.DOMINANT][0].organisms[neatnik.DOMINANT][0].score

        print(f"Cycle: {self.cycle_counter}  ∣  Nmbr. Species: {self.species_counter}  ∣  Max. Fitness: {max_score:.20f}", end="\n", flush=True)

        return

    def execute(self) -> None:
        """ Executes additional operations across all MPI ranks. """

        self.cycle_counter += 1
        self.species_counter = experiment.genus.size([0,1])

        return


experiment = AR1()
experiment.initialize()
experiment.run()

if experiment.MPI_rank == 0:
    organism = experiment.genus.species[neatnik.DOMINANT][0].organisms[neatnik.DOMINANT][0];
    p.dump(organism.data(), open('./ar1.p', 'wb'))
    print(organism.data())

experiment.finalize()