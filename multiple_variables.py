from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('training_data2.txt',names=['feet', 'rooms', 'price']).astype('float64')

# mean normalization
max_feet = data['feet'].max()
avg_feet = data['feet'].mean()
max_rooms = data['rooms'].max()
avg_rooms = data['rooms'].mean()
for row in data.values:
    row[0] = (row[0] - avg_feet) / max_feet
    row[1] = (row[1] - avg_rooms) / max_rooms
print('Normalized data:')
print(data)

# suppose that every row has format x1,x2,x3,x4,...,xn,y
# we don't subtract one, because we'll need a bias node
total_variables = len(data.values[0])
thetas = [0] * total_variables
alfa = 0.01

# hypothesis
def h(variables):
    sum = 0
    for i in range(len(variables)):
        # we use i + 1 becaise of bias node
        sum += variables[i] * thetas[i + 1]
    return thetas[0] + sum

def cost_function(data):
    return (1/len(data.values) * 2) * sum([(h(row[:-1]) - row[-1])**2 for row in data.values])

cost = cost_function(data)
print('First cost function = %s' % cost)

# learning by gradient descent
print('Start learning...')
for k in range(1500):
    new_thetas = [0] * total_variables
    for j in range(len(thetas)):
        # bias node
        if j == 0:
            errors = [h(row[:-1]) - row[-1] for row in data.values]
        else:
            errors = [(h(row[:-1]) - row[-1]) * row[j-1] for row in data.values]
        new_thetas[j] = thetas[j] - alfa * (1/len(data.values)) * sum(errors)
    thetas = new_thetas
    cost = cost_function(data)
    # print('=============================')
    # print('New cost = %s' % cost)

print('Finished learning!')
print('New cost function = %s' % cost_function(data))
print('Calculated parameters: %s' % thetas)

print('Check results on the trainig set:')
for row in data.values:
    predicted = h(row[:-1])
    print('=============================')
    print('Predicted: %s' % predicted)
    print('Real value: %s' % row[-1])
