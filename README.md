# 767-final-IR
COMP 767 final project

Alex Hoffman and Nikhil Podila

McGill University

We created a Python implementation of importance resampling algorithm from [Importance Resampling for Off-Policy Prediction](https://arxiv.org/pdf/1906.04328.pdf "Title") 
 
We also experimented with the addition of prioritized experience replay to the resampling algorithm

The code requires the following packages: numpy, gym, tensorflow, matplotlib. These can be installed with pip install or conda install if you use anaconda.
Running the file "OffPolicyAgent_testing.py" will produce plots depending on which functions are commented out at the bottom of the file. Hyperparameters are set in the body of the file. Experiment settings are set in the test functions (learning rates for the lr sweep, number of updates, steps per update, batch size). Feel free to raise an issue if you are having trouble navigating the code!

