#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
x = student_grades
y
ax = plt.subplot(111)
w = 0.1
ax.bar(x, y, width=w, color='b', align='edge')

"""graph design"""
plt.axis([0, 100, 0, 30])
plt.ylabel('Number of Students')
plt.xlabel('Grades')
plt.title('PROJECT A')
plt.show()
