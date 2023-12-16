import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from itertools import product

# Constants
I = range(1, 3)  # Suppliers
J = range(1, 4)  # Distribution centers (DC)
L_m = range(1, 5)  # Transportation modes
T = range(1, 3)  # Periods
G = range(1, 3)  # Products
K_levels = range(2, 4)  # Service levels
A = range(1, 6)  # Customers
F = (1, 2, 3)  # Available facilities
NF = (1, 3)  # Non-available facilities

# Parameters initialization
NL_lt = {(l, t): random.randint(1, 10) for l in L_m for t in T}
theta_k = {k: random.uniform(0, 1) for k in K_levels}
beta_it = {(i, t): random.uniform(0, 1) for i in I for t in T}
cL_l = {l: random.randint(100, 1000) for l in L_m}
fL_l = {l: random.randint(500, 1000) for l in L_m}
Cn_ja = {(j, a): random.randint(50, 200) for j in J for a in A}
Cs_jt = {(j, t): random.randint(50, 100) for j in J for t in T}
fo_jt = {(j, t): random.uniform(0, 1) for j in J for t in T}
Csh1_ja = {(j, a): random.randint(50, 200) for j in J for a in A}
Csh2_ijtg = {(i, j, t, g): random.randint(50, 200) for i in I for j in J for t in T for g in G}
Ch_jtg = {(j, t, g): random.randint(50, 200) for j in J for t in T for g in G}
w_ajtg = {(a, j, t, g): random.randint(800, 1400) for a in A for j in J for t in T for g in G}
gr_jg = {(j, g): random.randint(50, 200) for j in J for g in G}
Nv_jg = {(j, g): random.randint(20, 50) for j in J for g in G}
fE_jt = {(j, t): random.randint(50, 200) for j in J for t in T}
OB_j = {j: random.randint(1000, 100000) for j in J}
ax_j = {j: random.randint(500, 1000) for j in J}
tx_j = {j: random.randint(1500, 3000) for j in J}

# Integer Variables Ranges
N_ijtg = {(i, j, t, g): random.randint(10, 20) for i in I for j in J for t in T for g in G}
NS_jtg = {(j, t, g): random.randint(10, 100) for j in J for t in T for g in G}
NAS_jtg = {(j, t, g): random.randint(10, 50) for j in J for t in T for g in G}
HV_jgt = {(j, g, t): random.randint(100, 200) for j in J for g in G for t in T}
NG_jagt = {(j, a, g, t): random.randint(1, 10) for j in J for a in A for g in G for t in T}

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)
# Initialize DEAP toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("attr_int", random.randint, 1, 200)
# Define problem-specific parameters
NUM_Y = len(list(product(I, J, T, G)))
NUM_X = len(list(product(J, A, G, K_levels)))
NUM_K = len(list(product(J, A, L_m, T, K_levels)))
NUM_S = len(list(product(J, T, G)))
NUM_AS = len(list(product(J, T, G)))
NUM_DO = len(list(product(J, T)))

NUM_ATTRS = NUM_Y + NUM_X + NUM_K + NUM_S + NUM_AS + NUM_DO
def create_individual():
    # Initialize Y, K, S, AS, DO randomly
    Y = [random.randint(0, 1) for _ in range(NUM_Y)]
    K = [random.randint(0, 1) for _ in range(NUM_K)]
    S = [random.randint(0, 1) for _ in range(NUM_S)]
    AS = [random.randint(0, 1) for _ in range(NUM_AS)]
    DO = [random.randint(0, 1) for _ in range(NUM_DO)]

    # Initialize X with zeros
    X = [0] * (len(J) * len(A) * len(G) * len(K_levels))

    # Ensure at least one allocation for each combination of customer, product, and service level
    for a_idx, a in enumerate(A):
        for g_idx, g in enumerate(G):
            for k_idx, k in enumerate(K_levels):
                selected_facility = random.choice(list(F) + list(NF))
                idx = a_idx * len(G) * len(K_levels) * len(J) + g_idx * len(K_levels) * len(J) + k_idx * len(J) + selected_facility - 1
                X[idx] = 1  # Allocate the selected facility

    return creator.Individual(Y + X + K + S + AS + DO)

def create_individual():
    individual = [random.randint(0, 1) for _ in range(NUM_ATTRS)]

    # Ensure at least one allocation for each combination of customer, product, and service level
    for a in A:
        for g in G:
            for k in K_levels:
                # Randomly choose between available facilities (J) and non-available facilities (NF)
                facilities = J if random.random() < 0.5 else NF
                selected_j = random.choice(facilities)
                
                # Set the allocation for the selected facility
                idx = calculate_index(NUM_Y, a, g, k, selected_j)
        return individual
                
def calculate_index(num_y, a, g, k, j):
    a_idx = A.index(a)
    g_idx = G.index(g)
    k_idx = K_levels.index(k)
    j_idx = J.index(j) if j in J else NF.index(j) + len(J)  # Adjust for NF index
    return num_y + a_idx * len(G) * len(K_levels) * len(J) + g_idx * len(K_levels) * len(J) + k_idx * len(J) + j_idx


def convert_individual(individual):
    # Extracting parts of the individual corresponding to different variables
    Y = individual[:NUM_Y]
    X = individual[NUM_Y:NUM_Y + NUM_X]
    K = individual[NUM_Y + NUM_X:NUM_Y + NUM_X + NUM_K]
    S = individual[NUM_Y + NUM_X + NUM_K:NUM_Y + NUM_X + NUM_K + NUM_S]
    AS = individual[NUM_Y + NUM_X + NUM_K + NUM_S:NUM_Y + NUM_X + NUM_K + NUM_S + NUM_AS]
    DO = individual[NUM_Y + NUM_X + NUM_K + NUM_S + NUM_AS:]

    # Convert these parts into dictionary format for easier processing in constraints and objectives
    Y_dict = {key: Y[idx] for idx, key in enumerate(product(I, J, T, G))}
    X_dict = {key: X[idx] for idx, key in enumerate(product(J, A, G, K_levels))}
    K_dict = {key: K[idx] for idx, key in enumerate(product(J, A, L_m, T, K_levels))}
    S_dict = {key: S[idx] for idx, key in enumerate(product(J, T, G))}
    AS_dict = {key: AS[idx] for idx, key in enumerate(product(J, T, G))}
    DO_dict = {key: DO[idx] for idx, key in enumerate(product(J, T))}

    return Y_dict, X_dict, K_dict, S_dict, AS_dict, DO_dict

def objective_function_1(Y, X, K, S, AS, DO, N_ijtg, HV_jgt, NS_jtg, NAS):
    total_cost = 0

    # Cost component: Fixed cost for using transportation mode and transportation cost
    for t in T:
        for j in J:
            for a in A:
                for l in L_m:
                    for k in K_levels:
                        key = (j, a, l, t, k)
                        if key in K:
                            total_cost += fL_l[l] * Cn_ja[j, a] * K[key] * NL_lt[(l, t)]

    # Cost component: Fixed cost of opening DC
    for t in T:
        for j in J:
            key = (j, t)
            if key in DO:
                total_cost += fo_jt[key] * DO[key]

    # Cost component: Cost of holding in DC
    for j in J:
        for t in T:
            for g in G:
                key = (j, g, t)
                if key in HV_jgt:
                    total_cost += HV_jgt[key] * sum(w_ajtg[a, j, t, g] for a in A if (a, j, t, g) in w_ajtg)

    # Cost component: Shipping cost from supplier to DC and product price
    for i in I:
        for j in J:
            for t in T:
                for g in G:
                    for key in Y:
                        if Y[key] == 1:
                            total_cost += Csh2_ijtg[key] * N_ijtg[key] * (1 - beta_it[(i, t)])
                            total_cost += N_ijtg[key] * gr_jg[j, g]

    # Cost component: Cost of scarcity for DC and fixed cost for employing additional capacity
    for j in J:
        for g in G:
            for t in T:
                key = (j, t, g)
                if key in S:
                    total_cost += NS_jtg[key] * Cs_jt[(j, t)] * S[key]
                if key in AS:
                    total_cost += fE_jt[(j, t)] * NAS_jtg[key] * AS[key]

    # Cost component: Shipping cost from DC to customer
    for j in J:
        for a in A:
            for l in L_m:
                for t in T:
                    for k in K_levels:
                        for g in G:
                            key = (j, a, g, k)
                            if key in X:
                                total_cost += Csh1_ja[(j, a)] * w_ajtg[(a, j, t, g)] * X[key]

    return total_cost

# Objective Function 2: Minimize supply chain disruption risk (Z2)
def objective_function_2(Y, X):
    risk = 0

    # Risk component: Disruption risk at DC (j ∈ F)
    for i in I:
        for j in F:  # Available facilities
            for t in T:
                for g in G:
                    for k in K_levels:
                        key = (i, j, t, g)
                        if key in Csh2_ijtg and key in Y:
                            risk += Csh2_ijtg[key] * theta_k[k] * Y[key]

    # Risk component: Disruption risk at non-available facilities (j ∈ NF)
    for i in I:
        for j in NF:  # Non-available facilities
            for t in T:
                for g in G:
                    for k in K_levels:
                        key = (i, j, t, g)
                        if key in Csh2_ijtg and key in Y:
                            risk += Csh2_ijtg[key] * theta_k[k] * (1 - theta_k[k]) * Y[key]

    # Risk component: Disruption in shipping from DC to customer
    for j in J:
        for a in A:
            for g in G:
                for k in K_levels:
                    for t in T:
                        key = (j, a, g, k)
                        if key in Csh1_ja and key in X:
                            if j in F:
                                risk += Csh1_ja[key] * w_ajtg[a, j, t, g] * theta_k[k] * X[key]
                            elif j in NF:
                                risk += Csh1_ja[key] * w_ajtg[a, j, t, g] * theta_k[k] * (1 - theta_k[k]) * X[key]

    return risk

# Evaluate function
def customer_allocation_constraint_1(X, J, A, G, K_levels, F, NF):
    for a in A:
        for g in G:
            for k in K_levels:
                sum_X_J = sum(X.get((j, a, g, k), 0) for j in J)
                sum_X_NF = sum(X.get((j, a, g, k), 0) for j in NF)

                # Allow for some combinations to not have any allocation
                # Optionally, print a warning for such cases
                if sum_X_J + sum_X_NF == 0:
                    print(f"Warning: No allocation for Customer {a}, Product {g}, Service Level {k}")
                    continue  # Continue to the next combination

    return True  # All constraints satisfied


def check_constraints(individual):
    # Convert individual into format suitable for constraint checking
    Y_dict, X_dict, K_dict, S_dict, AS_dict, DO_dict = convert_individual(individual)
    if not customer_allocation_constraint_1(X_dict, J, A, G, K_levels, F, NF):
        print("Debug: Violation in customer_allocation_constraint_1")
        return False
    
    return True 

def create_valid_individual():
    for _ in range(10000):  # Increased number of attempts
        individual = create_individual()
        if check_constraints(individual):
            return individual
    raise Exception("Failed to create a valid individual after many attempts")

def evaluate(individual):
    Y, X, K, S, AS, DO = convert_individual(individual)

    # Check constraints
    if not check_constraints (individual):
    #(Y, X, K, S, AS, DO, J, A, G, L_m, T, K_levels, I, NF, w_ajtg, tx_j, ax_j, NAS_jtg, HV_jgt, N_ijtg):
        return 10000, 10000  # Large penalty for constraint violation

    # Calculate objectives
    obj1 = objective_function_1(Y, X, K, S, AS, DO, N_ijtg, HV_jgt, NS_jtg, NAS_jtg)
    obj2 = objective_function_2(Y, X)

    return obj1, obj2
# Initialize DEAP toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("attr_int", random.randint, 1, 200)
toolbox.register("individual", tools.initIterate, creator.Individual, create_valid_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# Register genetic operators
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selNSGA2)



# Main optimization algorithm
def main():
    NUM_INDIVIDUALS = 10
    population = toolbox.population(n=NUM_INDIVIDUALS)
    num_generations = 400
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    print("Evaluating initial population...")
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    for gen in range(num_generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = toolbox.select(offspring + population, len(population))
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        #print(logbook.stream)

    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    z1_values = []
    z2_values = []
    for ind in pareto_front:
        Y, X, K, S, AS, DO = convert_individual(ind)
        print("Y: ", Y)
        print("X: ", X)
        print("K: ", K)
        print("S: ", S)
        print("AS: ", AS)
        print("DO: ", DO)
        print("N_ijtg: ", N_ijtg)
        print("HV_jgt: ", HV_jgt)
        print("NS_jtg: ", NS_jtg)
        print("NAS_jtg: ", NAS_jtg)
        
        z1, z2 = ind.fitness.values
        z1_values.append(z1)
        z2_values.append(z2)
        print("Objective 1 (Z1):", z1)
        print("Objective 2 (Z2):", z2)
        print("-----------------------")

    # Find and print overall max and min of Z1 and Z2
    max_z1, min_z1 = max(z1_values), min(z1_values)
    max_z2, min_z2 = max(z2_values), min(z2_values)
    print("Max Z1:", max_z1, "Min Z1:", min_z1)
    print("Max Z2:", max_z2, "Min Z2:", min_z2)

    # Plotting the Pareto front
    plt.scatter(z1_values, z2_values, c="red")
    plt.title("Pareto Front")
    plt.xlabel("Objective 1 (Z1)")
    plt.ylabel("Objective 2 (Z2)")
    plt.show()

if __name__ == "__main__":
    main()
