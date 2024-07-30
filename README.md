# Geostatistics
Code containing various geostatistical tools that are useful for PET.

## Installation
Clone repository and install with pip:

```sh
pip install -e .
```
## Examples
```python
from geostat.decomp import Cholesky
import numpy as np

stat = Cholesky()
nx = 3
mean = np.array([1., 2., 3.])
var = np.array([10., 20., 30.])
ne = 2
cov = stat.gen_cov2d(x_size=nx, y_size=1, variance=var,
                                  var_range=1., aspect=1., angle=0., var_type='sph')
sample = stat.gen_real(mean, cov, ne)
```
