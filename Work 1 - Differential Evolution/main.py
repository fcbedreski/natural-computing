import math
from scipy import optimize
from scipy.optimize import NonlinearConstraint
import numpy as np


#Reading test cases to store in variables
file = open("Caso_13geradores_econ_ponto_valvula.txt", "r")

generators = int(file.readline()) #read 1st line
file.__next__ #go to 2nd

potency = int(file.readline()) #read 2nd
file.__next__ #do to 3rd


#Reading the parameters that are in the sequence, in the file
def reading_params():
    aux_list = []
    for i in range(generators):
        aux_list.append(file.readline())
        file.__next__
    
    return aux_list

#reading P_min
P_min = reading_params()

#reading P_max
P_max = reading_params()

#reading the a's 
a = reading_params()

#reading the b's 
b = reading_params()

#reading the c's 
c = reading_params()

#reading the d's 
d = reading_params()

#reading the e's 
e = reading_params()

#reading the f's 
f = reading_params()

#---- Creating bounds vector ----
bounds = []
for i in range(generators):
    tup = (P_min[i], P_max[i])
    bounds.append(tup)


#Objective Function - needs to be placed in a loop, because we're dealing with a vector of data
#x = P_i 

def objective_f(x):
    aux_sum = 0

    # a_i * P_i ** 2 + b_i * P_i + c_i 
    # + |e_i * sine(f_i * (P^min_i - P_i))|
    for i in x:
        aux_sum += math.pow(a[i] * x[i], 2) + b[i] * x[i] + c[i] 
        + abs(e[i] * math.sin(f[i] * P_min[i] - x[i]))

    return aux_sum

def produced_f(x = ()):
    aux_sum = 0

    # P_i 
    for i in x:
        aux_sum += x[i]

    return aux_sum

#NLC - Non Linear Constraints 
# sum_of_i(P_i) - P_D = 0
nlc = NonlinearConstraint(produced_f() - potency, -0.01, 0.01)

result = optimize.differential_evolution(objective_f, bounds, args=(), 
    strategy='best1bin', #Can be changed to check new results
    maxiter=1000, #Can be changed to check new results
    popsize=50, #Can be changed to check new results
    tol=0.01, 
    mutation=(0.5, 1), 
    recombination=0.7, 
    seed=None, callback=None, 
    disp=False, 
    polish=False, #Was changed to false
    init='latinhypercube', 
    atol=0, updating='immediate', 
    workers=1, #Means it's parallelizable
    constraints=(nlc), #Non Linear Contraints
    x0=None)


#Printing the results
print('Result x: ')
print(result.x)
print('/n')
