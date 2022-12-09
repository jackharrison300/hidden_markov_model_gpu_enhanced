Run program on Palmetto desktop with V100 GPU (requires desktop for visual output)

To install modules and dependencies
- `module load cuda/11.6.2-gcc/9.5.0`
- `module load anaconda3/2022.05-gcc/9.5.0`
- `pip install pycuda`
- `pip install numpy`

To run GPU program
- `cd new_implementation` from main folder
- `python hmm.py`

To run CPU program
- `cd orig_implementation` from main folder
- `python hmm.py`

Once the program is running, to begin the Hidden Markov Model execution, click "Run" in the GUI
NOTE: only the "Exact Inference" implementation, which is the default, is pertinent to this project.

Expected output:
- Execution time for 100 iterations printed to command line
- Visual output of tracker