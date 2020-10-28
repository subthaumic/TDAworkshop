# Exercise 5
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams


# U
x = [0, 0]
y = [-1, 11]
z = [9, 1]
w = [11, 10]

data = np.array([x, y, z, w])
plt.scatter(data[:, 0],data[:, 1])
plt.show()

# The Rips filtration of U  is as follows (deduced either from explicit calculation of distances or by visual inspection)
# R_0 = { '0-simplices': [x, y, z, w] }
# R_1 = { '0-simplices': [x, y, z, w], '1-simplices': [x,z] }
# R_2 = { '0-simplices': [x, y, z, w], '1-simplices': [[x,z], [z,w]] }
# R_3 = { '0-simplices': [x, y, z, w], '1-simplices': [[x,z], [z,w], [x,y]] }
# R_4 = { '0-simplices': [x, y, z, w], '1-simplices': [[x,z], [z,w], [x,y], [x,w]] }   <-  this is where the 1-simplices form a cycle, which is not yet filled in
# R_5 = { '0-simplices': [x, y, z, w], '1-simplices': [[x,z], [z,w], [x,y], [x,w], [y,z]], '2-simplices': [[x,y,z], [x,z,w]] } <- adding the diagonal edge [y,z] forms triangles, which are automatically filled in in the rips complex
# ... <- higher steps in the filtration are not interesting anymore, because the complex in the previous step was already contractible to a point.


intervals = ripser(data)['dgms']
plot_diagrams(intervals,0)
plt.show()
plot_diagrams(intervals,1)
plt.show()


# W
x = [0,10]
y = [2.5,1]
z = [5.5,5]
w = [7.5, 0]
v = [10,9]

data = np.array([x,y,z,w,v])
plt.scatter(data[:,0],data[:,1])
plt.show()

intervals = ripser(data)['dgms']
plot_diagrams(intervals,0)
plt.show()
plot_diagrams(intervals,1)
plt.show()


# The two examples show that four points can be enough to form a bar in H_1. However, in general any fifth point that lies in between these four points will prevent such a bar to form.
