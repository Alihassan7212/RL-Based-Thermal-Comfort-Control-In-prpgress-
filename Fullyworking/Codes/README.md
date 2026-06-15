I have broken down the steps to train and test the RL model.

------ SETUP --------
1. git clone the boptest repository.
"HTTPS: https://github.com/ibpsa/project1-boptest.git"

2. Install docker on your setup if not already installed 

3. Go into the boptest repo and run the following command on the terminal:
"docker compose up web worker provision --scale worker=8"

4. Now the docker is running. Make sure the API endpoint is correct (it is 8000 by default)

5. Now run the requirements.txt file " pip install -r requirments.txt"

----- The setup is complete -------- (This is a one time activity)

------ TRAINING ---------

Simply run the "train_rl.py" file located inside "Fullyworking_2_reward repo"

The trained model will be stored in trained_model repo.

------ Training Complete --------

------ EVALUATING THE MODEL -----------

In order to see the results and compare them, run the script from "Results_generator" repo
It will produce graphs along with numerical results.
These results are stored inside the trained_model repo as well.
