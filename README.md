# forecasting

We study the language of good and bad forecasts. Generally, a question is asked and people make forecast between a _startdate_ and an _enddate_. The _enddate_ is always before the date in which the answer is known or sure. At some point between _stratdate_ and _enddate_, the majority of the forecasts (almost always) agree with the correct answer.

# Library versions

Hugging Face Transformer library: 4.5.1
Pytorch: Most Recent Version should work

# Code Details

Model.py contains 2 model architectures implemented using HuggingFace: The LSTM that did not concatenate question information (LSTM_Model), 
and the LSTM that did concatenate question information (LSTM_Model_With_Question)

Utils.py contains the code for processing the actual GJO Questions (which are found in data/), and creating the train/dev/test splits. The actual train/dev/test 
splits I used are found in questions.save. 

Train.py contains the code for initializing all the hyperparameters of the model and the code for the training and testing loops/printing out the results. 