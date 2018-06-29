from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

theta0 = 0
theta1 = 0
alfa = 0.01

# hypothesis
def h(x,theta0,theta1):
    return theta0 + theta1 * x

# our cost function
def cost_function(data):
    cost = 0
    return sum([(h(row[0], theta0, theta1) - row[1])**2 for row in data])/(len(data) * 2)

data = pd.read_csv('training_data1.txt')
cost = cost_function(data.values)
print('Cost function = %f' % cost)

difference = 99999999
# repeat until convergence
while difference > 10**(-15):
    errors0 = [h(row[0], theta0, theta1) - row[1] for row in data.values]
    errors1 = [(h(row[0], theta0, theta1) - row[1]) * row[0] for row in data.values]
    new_theta0 = theta0 - alfa * (1/len(data.values)) * sum(errors0)
    new_theta1 = theta1 - alfa * (1/len(data.values)) * sum(errors1)
    theta0 = new_theta0
    theta1 = new_theta1
    new_cost = cost_function(data.values)
    difference = cost - new_cost
    cost = new_cost
    print('New cost = %s' % cost)

print('Finished learning')
print('Thetas %s' % [theta0,theta1])

# test on training data(know that's bad)
print('Check result on the training set:')
for row in data.values:
    print('========================')
    predicted = h(row[0],theta0,theta1)
    print('Predicted value %s' % predicted)
    print('Real value %s' % row[1])
