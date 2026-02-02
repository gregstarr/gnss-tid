| Data variables    | Dimensions                | Description                                           |
| :---              | :---                      |  ---:                                                 |
| `image`           | `(time, y, x)`            | focused TID image (TEC)                               |
| `density`         | `(time, y, x)`            | number of IPPs within a radius of each image pixel    |
| `patch`           | `(time, py, px, ky, kx)`  | abs(FFT)^2 on each image patch                        |
| `Fx`              | `(time, py, px)`          | x wavenumber of maximum FFT component (1/km)          |
| `Fy`              | `(time, py, px)`          | y wavenumber of maximum FFT component (1/km)          |
| `F`               | `(time, py, px)`          | magnitude of maximum FFT component                    |
| `height`          | `(time)`                  | focused IPP height (km)                               |
| `objective`       | `(time)`                  | objective function value                              |
| `wavelength`      | `(time)`                  | wavelength of TID (km)                                |
| `offset`          | `(time)`                  | phase offset of TID (km)                              |
| `phase`           | `(time)`                  | phase offset of TID (rad)                             |
| `center`          | `(ci)`                    | TID center location (km)                              |

| Coordinates   | Description                                   |
| :---          | ---:                                          |
| `x`           | x coordinate in local cartesian system (km)   |
| `y`           | y coordinate in local cartesian system (km)   |
| `time`        | date/time                                     |
| `kx`          | x wavenumber (FFT bin) for each patch (1/km)  |
| `ky`          | y wavenumber (FFT bin) for each patch (1/km)  |
| `px`          | x coordinate of patch center (km)             |
| `py`          | y coordinate of patch center (km)             |
