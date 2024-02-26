# vu-ml-project

## Project setup

1. Install poetry [here](https://python-poetry.org/docs/)

2. Install dependencies
```bash
poetry install
```

3. Select the venv as the kernel in jupyter notebook
    - In `src/` there is `testing.ipynb` which is just the main testing file for now 
    - Open the notebook and in the top right corner select the kernel 
    - If the kernel doesn't show in the drop down you should be able to click on create kernel and then select the venv

4. Run the cell, it should output `model1`, this should confirm that the environment is set up correctly