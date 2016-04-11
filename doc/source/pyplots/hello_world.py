import numpy as np
import pyny3d.geoms as pyny

# Geometry creation
## Polygons by their vertices
base = np.array([[0,0], [10,0], [10,10], [0,10]]) # Base square on the floor
pillar = np.array([[4,4,8], [6,4,8], [6,6,8], [4,6,8]]) # Top obtacle polygon (to extrude)

## Obstacle
place = pyny.Place(base)
place.add_extruded_obstacles(pillar)
space = pyny.Space(place)

# Shadows
S = space.shadows(init='auto', resolution='high')

# Viz
S.viz.exposure_plot()

