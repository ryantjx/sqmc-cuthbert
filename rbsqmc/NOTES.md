220826
- wrote the script for optimizing on GPU
- wrote the train_model which trains the filter sequentially
- seems to be getting better results with log_marginal optimization

230826
- ran without inverse wishart prior, seems to be converging and doing better
- completed sqmc model
- utilized qmc with rbpf model
- wrote rbsqmc model

240826
- add time series filter for states with games 
- need to run a script to generate model with different hyperparameters