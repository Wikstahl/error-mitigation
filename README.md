# error-mitigation

[![Tests](https://github.com/Wikstahl/error-mitigation/actions/workflows/python-app.yml/badge.svg)](https://github.com/Wikstahl/error-mitigation/actions/workflows/python-app.yml) [![DOI](https://zenodo.org/badge/442167639.svg)](https://zenodo.org/badge/latestdoi/442167639)


## Installation <a name="installation"></a>
  Clone the files
  ```
  git clone git@github.com:Wikstahl/error-mitigation.git
  cd error-mitigation
  ```
  Recommended: Create a new virtual environment using **conda** or **venv**
  ```
  python -m venv .venv
  ```
  activate it using
  ```
  source .venv/bin/activate
  ```
  install the required packages
  ```
  python -m pip install -r requirements.txt
  ```
  install the module
  ```
  python -m setup.py
  ```

## Examples <a name="examples"></a>
Examples of virtual distillation applied to qaoa states and thermal states are given in the folder `examples`

## Usage <a name="usage"></a>
The `src` folder contains the scripts used for producing the results. The *qaoa* folder produces the virtual distillation for variational states results, and *thermal* produces virtual distillation applied to thermal states results.
