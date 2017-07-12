# H5HutAccessor
A utility Python class for accessing
[H5Hut](http://vis.lbl.gov/Research/H5hut/InternalLayout.html) files
generated/consumed by [OPAL](https://gitlab.psi.ch/OPAL/src/wikis/home)

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

### H5Hut/H5Part file structure
Since this isn't particularly obviously documented by OPAL, here's some
unorganized notes on the file structure consumed/produced by OPAL. Note that I
don't yet use some of OPAL's more sophisticated features like tracking multiple
bunches, so that some of the shapes below may be too specific.

    <HDF5 file root>
    * `Resonance Frequency(Hz)` [attr], shape (1,) ndarray of one element giving RF frequency
    * `Step#0` [group]
      * ... (see below)
      * `Block` [group]
        * `Efield` [group]
          * `__Origin__` [attr], shape (3,) ndarray of grid origin (**in mm per `ascii2h5block_asgic.cpp`**)
          * `__Spacing__` [attr], shape (3,) ndarray of grid spacing (**in mm per `ascii2h5block_asgic.cpp`**)
          * `0` [dataset] - x component of field (TODO: shape)
          * `1` [dataset] - y component of field (TODO: shape)
          * `2` [dataset] - z component of field (TODO: shape)
        * `Bfield` [group]
          * Same structure as `Efield`
    * ...
    * `Step#N` [group]
      * `TIME` [attr], shape (1,) ndarray: step time (sec)
      * `SPOS` [attr], shape (1,) ndarray: reference (not sure if/how this relates to `spos-ref, REFZ`) `s` position (m)
      * `RMSX` [attr], shape (3,) ndarray: rms beam size (`sigma_x, sigma_y, sigma_z`) (m)
      * `centroid` [attr], shape (3,) ndarray: beam centroid position (m)
      * `x` [dataset], shape (Nparticles,) ndarray of particle x positions (m)
      * `y` [dataset], shape (Nparticles,) ndarray of particle y positions (m)
      * `z` [dataset], shape (Nparticles,) ndarray of particle z positions (m)

N.B. ALL of the structure of `Block` is **required** for specifying fields with
H5Part files, even if one of the fields is zero everywhere etc!

### Acknowledgements
Thanks to Daniel Winklehner for providing an example
program (`ascii2h5block_asgic.cpp`) for creating an H5Part fieldmap file.
