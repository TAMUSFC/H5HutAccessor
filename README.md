# H5HutAccessor
A utility class in Python for accessing HDF5 files written by [OPAL](https://amas.psi.ch/OPAL) in the [H5Hut](https://amas.psi.ch/H5hut/) format.

## Example Usage
```python
from H5HutAccessor import H5HutAccessor
import numpy as np

h = H5HutAccessor('path/to/my/file.h5')

# Simulation steps can be accessed by direct indexing. 
# Note that the indices here are **not** the step number as reported by OPAL!
firststep = h[0]
laststep = h[-1]
firsttensteps = h[0:10]

# Step data are available with dot notation
print("On first step, r = {coords}".format(coords=(firststep.x, firststep.y, firststep.z)))
print("First ten steps range from {tmin} s to {tmax} s".format(tmin=firsttensteps.t.min(), tmax=firsttensteps.t.max()))

# Arrays for all steps are available directly on the H5HutAccessor object
r = np.sqrt(h.x**2 + h.y**2)
p = np.sqrt(h.px**2 + h.py**2 + h.pz**2)
print("Imported file with {} steps".format(len(h)))
print("Minimum and maximum radii in XY plane: rmin = {rmin} m, rmax = {rmax} m".format(rmin=r.min(), rmax=r.max()))
print("Starting momentum: {pstart}, Final momentum: {pfinal}".format(p[0], p[-1]))
```

The `with` keyword is also supported:

```python
from H5HutAccessor import H5HutAccessor
import numpy as np

with H5HutAccessor('path/to/my/file.h5') as h:
  print("Imported file with {} steps".format(len(h)))
```
