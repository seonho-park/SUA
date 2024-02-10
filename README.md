# Optimization Formulation
- please refer to the [PDF](./doc/formulation.pdf) file.

# Testing Environment
- Tested on MacBook Pro with M1 MAX
- Python 3.10.13

# Dependency
- scipy 1.11.4
- pandas 2.1.4
- numpy 1.26.3
- pyomo 6.7.0
- glpk 5.0
- highs 1.5.3

# Installation with Conda
- in bash, create the following conda environment
    ```bash
    conda env create -f environment.yml
    ```

# Execution
- after locating input_SUA.xlsx file in the root, please execute the following.
    ```bash
    python main.py --mu 0.1 --ns 100
    ```
    
- argument `mu` is the ratio of aggregate surplus quantity across all products over the total estimated demands.
- argumeent `ns` is the number of scenarios (samples) for representing the stochasticity of the demand
