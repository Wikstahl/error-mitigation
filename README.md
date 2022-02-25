# error-mitigation
Code for reproducing the results in arxiv

[![Tests](https://github.com/Wikstahl/error-mitigation/actions/workflows/python-app.yml/badge.svg)](https://github.com/Wikstahl/error-mitigation/actions/workflows/python-app.yml)

# Table for contents
  1. [Installation](#installation)
  2. [Examples](#examples)
  3. [Usage](#usage)

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
Examples of how the code is produced is given the the folder `examples`

## Usage <a name="usage"></a>
The `src` folder contain the scripts for reproducing the results. The *qaoa* folder is for reproducing the virtual distillation for variational state, and *thermal* is for reproducing virtual distillation applied to thermal states.
