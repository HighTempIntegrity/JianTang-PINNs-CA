# ThermoPINNs

This repository includes the Python package **ThermoPINNs**, which uses Physics-Informed Neural Networks (PINNs) to solve thermal problems in single-track laser powder bed fusion (LPBF).

## Source Code
To access the source code for ThermoPINNs, see the folder `pinns_code/ThermoPINNs/`. You can modify the `config.yaml` file and run `test.py` to train a PINNs model. A manual for each parameter in `config.yaml` is provided in `pinns_code/ThermoPINNs/config_function.py`.

## Dependencies
For package dependencies, refer to [Python on Euler](https://scicomp.ethz.ch/wiki/Python_on_Euler#python_gpu.2F3.11.2). In addition to this environment, install `sobol-seq` and `pyDOE` using:

```bash
pip install pyDOE sobol-seq
```

## Modifications for New Problems
For new problems, such as different materials or process conditions, modify pinns_code/ThermoPINNs/Paper_Equation_param.py as needed.
