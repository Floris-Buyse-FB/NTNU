# NTNU

## How to use

1. Create a virtual environment: `python3 -m venv .venv`
2. Activate venv: `source .venv/bin/activate`
3. Install requirements: `pip3 install -r requirements.txt`
4. Make sure 'main.sh' has execution permissions: `chmod +x main.sh`
5. Execute 'main.sh': `./main.sh`
6. Follow the instructions in the terminal

## If problems occur

### No csv file called 'clean_data.csv' in the data folder

- If there is no `clean_data.csv` file in the data folder download it from the [GBIF Website](https://doi.org/10.15468/zrlqok)
- Cleaning this file can be done by running the cel in the `notebooks/clean_csv.ipynb` notebook

### Path errors

- In the `scripts/classify_images.py` file, the Ultralytics runs directory can be updated by changing the constant `RUNS_DIR`
- Use an absolute path to be sure that the program can find the correct directory

<br>

- The other paths are relative and should not cause any problems

### Module errors

- If there are any module errors, make sure that the requirements are installed
- Running seperate scripts can cause Module errors
  - run them using: `python3 -m scripts.script_name` from the root directory
  - note: without the `.py` extension
  - any arguments can be found using the `-h` flag
- Note: it is recommended to run the `main.sh` script to avoid any errors