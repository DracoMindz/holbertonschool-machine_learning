#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

"""data format"""
bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.hist(student_grades, align='mid', bins=bin_edges, edgecolor='black')


"""graph design"""
plt.ylabel('Number of Students')
plt.xlabel('Grades')
plt.title('Project A')
plt.xlim(0, 100)
plt.xticks(bin_edges)
plt.ylim(0, 30)
plt.show()
