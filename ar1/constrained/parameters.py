import neatnik


neatnik.Parameters.random_seed = 9748581
neatnik.Parameters.generational_cycles = 100
neatnik.Parameters.population_size = 100
neatnik.Parameters.mutation_attempts = 10
neatnik.Parameters.spawning_attempts = 10
neatnik.Parameters.weight_bound = 1.0
neatnik.Parameters.perturbation_power = 2.0
neatnik.Parameters.splitting_priority = 2.0
neatnik.Parameters.initial_activation = neatnik.RELU
neatnik.Parameters.rejection_fraction = 0.3
neatnik.Parameters.stagnation_threshold = 15
neatnik.Parameters.compatibility_threshold = 5.0
neatnik.Parameters.compatibility_weights = [2., 2., 5.]
neatnik.Parameters.enabling_link = [1.0, 0.0]
neatnik.Parameters.altering_links = [0.0, 1.0]
neatnik.Parameters.altering_weight = [0.0, 0.2, 0.8]
neatnik.Parameters.adding_link = [0.5, 0.25, 0., 0.25, 0.]
neatnik.Parameters.enabling_node = [1., 0., 0.]
neatnik.Parameters.altering_nodes = [1., 0.]
neatnik.Parameters.altering_activation = [1., 0., 0., 0.]
neatnik.Parameters.adding_node = [1., 0., 0.]
neatnik.Parameters.assimilating_links = [0., 1.]
neatnik.Parameters.assimilating_nodes = [1., 0.]
neatnik.Parameters.assimilating_weight = [0.5, 0.5]
neatnik.Parameters.assimilating_activation = [1., 0.]
neatnik.Parameters.spawning_organism = [0.4, 0.6]
