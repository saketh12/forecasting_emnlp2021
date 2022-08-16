# forecasting
This is the code for the paper, [Written Justifications are Key to Aggregate Crowdsourced Forecasts](https://aclanthology.org/2021.findings-emnlp.355.pdf).

# Code Details
To install dependencies, pip install the libraries listed in requirements.txt.

Model.py contains 2 model architectures implemented using HuggingFace: The LSTM that did not concatenate question information (LSTM_Model), 
and the LSTM that did concatenate question information (LSTM_Model_With_Question)

Utils.py contains the code for processing the actual GJO Questions (which are found in data/), and creating the train/dev/test splits. The actual train/dev/test 
splits I used are found in questions.save. 

Train.py contains the code for initializing all the hyperparameters of the model and the code for the training and testing loops/printing out the results. 

To replicate, run utils.py then train.py.