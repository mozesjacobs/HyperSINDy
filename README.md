# HyperSINDy

Implementation of SINDy using a hypernetwork and amortized variational 
inference ("Bayes by Hypernet").

## Data
Generate the Lorenz data from the base of the repository by running 
python3 generate_lorenz.py

Generate the Rossler data from the base of the repository by running 
python3 generate_rossler.py

## Running model
To run the model, configure arguments in cmd_line. <br>
Then, run python3 main.py <br>
Results are saved in tensorboard