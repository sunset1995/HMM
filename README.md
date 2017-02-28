# HMM

Implement Discrete Hidden Markov Model.


## Quick Start
Run `pip3 install -r requirements.txt` to install the dependency.  
Copy or create a soft link to the directory `hmm` to the diretory you work on.  
Here is an quick example:
```
from hmm.hmm import DiscreteHMM

# Setting model's initial parameter
A = (
	(0.1, 0.1, 0.8),
	(0.1, 0.1, 0.8),
	(0.1, 0.1, 0.8), )
B = (
	(0.1, 0.9),
	(0.2, 0.8),
	(0.3, 0.7), )
pi = (1/3, 1/3, 1/3)

# Create an model
hmm = DiscreteHMM(3, 2, A=A, B=B, pi=pi)

# Train the model to a local maximal
# which better describes the given observations than original model
train_obs_seq = (0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0)
hmm.train(train_obs_seq, verbose=1)

# Show the model
hmm.show_model()

# Check whether each row sum to 1.0
print('checksum:', hmm.check_model())
print('=' * 20)

# Predict the probabilit distribution of the hidden state given a observation sequence
prediction = hmm.given((1, 0, 1, 0, 1, 0, 1))
print('Current hidden state:', prediction['forward'])
print('Most likely path    :', prediction['viterby'])
print('=' * 20)

# Continuing with previous given sequence and making prediction
for obs in (1, 0, 0, 1):
	prediction = hmm.given_more((obs, ))
	print('Current hidden state:', prediction['forward'])
	print('Most likely path    :', prediction['viterby'])
	print('=' * 20)
```


## Examples
- [whether forecast](./examples/weather_forecast/)
- [forex](./examples/forex/)


## Documents

### DiscreteHMM(N, M, A, B, pi)
- Counstructor of the hmm model
- N: _required_, _int_. Number of hidden states
- M: _required_, _int_. Number of observation symbol
- A: _optional_, _tuple of tuple_. State transition probability
	- Randomly generate one if not given
	- If the dimension is not `N by N` or each row is not sum to 1.0, Assertion error will be raised
- B: _optional_, _tuple of tuple_. Emmision probability
	- Randomly generate one if not given
	- If the dimension is not `N by M` or each row is not sum to 1.0, Assertion error will be raised
- pi: _optional_, _tuple_. Initial state probability
	- Randomly generate one if not given
	- If the dimension is not `1 by N` or the only one row is not sum to 1.0, Assertion error will be raised

After creating a model, you can use it to make some prediction (see [weather forecast viterby example](./examples/weather_forecast/viterbi.py)) or using some observation sequences to train it (see [weather forecast train example](./examples/weather_forecast/train.py)).  

### DiscreteHMM.train(obs_seq, itnum, eps, verbose):
- Using the given `obs_seq` to train the model
- Each member in `obs_seq` should between `0` and `M-1`
- obs_seq: _required_, _tuple_. The given observation sequence
- itnum: _optional_, _int_. Maximum # of iterations. Default is `100`
- eps: _optional_, _float_. Stop if the sum of difference of each entries (called delta) between two iteration is less than `eps`. Default is `0.01`
- verbose: _optional_, _int_ `0` shows nothing, `1` shows iteration count and delta, `2` show model after each iteration. Default is `0`

Call `train` multiple times with different `obs_seq` or `train` after some prediction is available.  

### DiscreteHMM.given(obs_seq)
- Start with `obs_seq`, coculate the hidden state probability distribution and viterby path given the sequence.
- Each member in `obs_seq` should between `0` and `M-1`
- obs_seq: _required_, _tuple_. The given observation sequence
- Return a `dict`
	- `forward`: a numpy array. Probability of each hidden state.
	- `viterby`: a list. The viterby path which is the most likely path generating the `obs_seq`

### DiscreteHMM.given_more(obs_seq)
- Like `given` but the result will base on previous given `obs_seq`
- You can calling `given_more` without calling `given`. Then the first time calling `given_more` is equal to `given`
- If you want to make model forget all previous `obs_seq`, call `given`

### DiscreteHMM.show_model()
- Show the model

### DiscreteHMM.check_model()
- Return whether each row sum to 1.0
