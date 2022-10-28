# CLIMPCA

Software to reconstruct missing data in climate datasets using PCA

## Dependencies
- python>=3.9.13
- tqdm>=4.64.1
- cuml>=22.10.00
- numpy>=1.23.4
- xarray>=2022.10.0
- netcdf4>=1.5.7
- setuptools=59.5.0

An Anaconda environment with all the required dependencies can be created using `environment.yml`:
```bash
conda env create -f environment.yml
```
To activate the environment, use:
```bash
conda activate climpca
```

## Installation

`climpca` can be installed using `pip` in the current directory:
```bash
pip install .
```

## Usage

```
climpca -f <input-file>
```

## License

`climpca` is licensed under the terms of the BSD 3-Clause license.
