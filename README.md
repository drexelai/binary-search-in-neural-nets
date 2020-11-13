# binary-search-in-neural-nets

Drexel AI's Fall Term research project on efficiently searching for accurate neural net architectures
## How to run

With python version over 3.6(I'm using 3.7.1), do
```
pip install requirements.txt
```
Then run
```
python run.py
```
There are some arguments that can be used. For example, to change epoch size, you can do
```
python run.py --epoch=200
```
The list of arguments can be seen at binary_search_networksbinary_search_parser.py

## Current features

- Gets data.
- Preprocesses data.
- Given input n, trains model. Prints out accuracy.
- Tests model. Prints out accuracy.

## Check list

Deadline: December 12, 2020


-[x] Pick a Dataset that can be generalized -> Using titanic dataset courtesy of https://www.openml.org/d/40945
-[] Implement linear search (O(N)) # need to train the model N times. 
-[] Implement binary search (O(log(N))) # need to train the model N times. 
-[] Determine trendline over two end points, determine the slope and determine side to get rid of
-[] Find the maximum value in a partially sorted array
-[x] Adapt the code so that user can pass n as input and run the entire pipeline (train + test + save)
-[x] Plot the loss and accuracy given a neural net for our problem 