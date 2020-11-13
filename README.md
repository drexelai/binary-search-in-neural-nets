# binary-search-in-neural-nets
Drexel AI's Fall Term research project on efficiently searching for accurate neural net architectures


## Run from Scratch
- First, download Python from here: https://www.python.org/downloads/windows/  
- Then, download get-pip.py from https://bootstrap.pypa.io/ \
`$ cd /path/to/downloadsfolder` \
`$ python get-pip.py` 
- Install Git from here: https://gitforwindows.org/ 
- Install requirements \
`$ pip install -U -r requirements.txt` 
- At this point, you should be able to run the neural network \
`$ python ann.py`  



## Included Files ##
- binary-model-1/ann.py
- binary-model-1/Churn_Modelling.csv
- binary-model-1/model.json
- binary-model-1/model.h5
- requirements.txt
- README.md
- binary-model-1/group1-shard1of1.bin (Keras model is converted to Tensorflow layers so that it could be imported and run on a web browser.)


Deadline: December 12, 2020

-[] Pick a Dataset that can be generalized
-[] Implement linear search (O(N)) # need to train the model N times. 
-[] Implement binary search (O(log(N))) # need to train the model N times. 
-[] Determine trendline over two end points, determine the slope and determine side to get rid of
-[] Find the maximum value in a partially sorted array
-[] Adapt the code so that user can pass n as input and run the entire pipeline (train + test + save)
-[] Plot the loss and accuracy given a neural net for our problem 