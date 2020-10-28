# Initialize Mapper
import kmapper as km
mapper = km.KeplerMapper()

# Import sample data (2 disjoint circles)
from sklearn import datasets
data, labels = datasets.make_circles(n_samples=5000, noise=0.03, factor=0.3)

# project the data, here: tautological projection to x,y coordinates
# you could for example try projection='sum' instead or choose one of the other preset options. see help(mapper.project)
projected_data = mapper.fit_transform(data, projection=[0, 1])

# Create dictionary called ’graph’ with nodes, edges and meta information
graph = mapper.map(projected_data, data, cover=km.Cover(n_cubes=10, perc_overlap=0.5))


mapper.visualize(graph, path_html="getting_started.html", title="description_of_choices")
