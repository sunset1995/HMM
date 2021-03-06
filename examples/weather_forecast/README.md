# HMM: Wheather Forecast Examples

There are three hidden variable: `sunny`, `foggy`, `rainy`  
And two observed variables: `yes`, `no`, indicating whether your friend worn his coat before going out.  
The data sequence `input.txt` is generated by below markov model:  

### Transition probability table
|       | sunny | foggy | rainy |
| ----- | ----: | ----: | ----: |
| sunny | 0.8 | 0.15 | 0.05 |
| foggy | 0.2 | 0.5 | 0.3 |
| rainy | 0.2 | 0.2 | 0.6 |

### Emmision probability table
| weather | P(coat=no ￨ weather) | P(coat=yes ￨ weather) |
| ------- | -------------------: | --------------------: |
| sunny | 0.9 | 0.1 |
| foggy | 0.7 | 0.3 |
| rainy | 0.2 | 0.8 |


## Viterbi Algorithm
Run `python3 viterbi.py` to see the result of _best explanation of the given observations_.  
In this case, the model is known. So encode the parameter to hmm directly.
```python
A = (
        (0.80, 0.15, 0.05),
        (0.20, 0.50, 0.30),
        (0.20, 0.20, 0.60),
    )
B = (
        (0.9, 0.1),
        (0.7, 0.3),
        (0.2, 0.8),
    )
pi = (0.5, 0.25, 0.25)
hmm = DiscreteHMM(len(hidden_var), len(observation_var), A=A, B=B, pi=pi)
```

To see the result of `argmax_{q[:]} P(O[:] | q[:])`
```python
viterby_path = hmm.given(obs_seq)['viterby']
```
which is the most likely weather sequence generating the observed sequence.


## Training the model
Run `python3 train.py` to see the trained model which best fit to the observed sequence.  
In this case, the model is unknown. The hard problem is that: Because of the fact that the training result only depend on the initial setting, how can we know which hidden variable id corresponding to which weather?  
In `train.py`, I assume the id corresponding to each hidden variable is:
```python3
hidden_var = {
    'sunny': 0,
    'foggy': 1,
    'rainy': 2,
    }
```
And set the inital emmision probability directly, expecting it will result to a local maximum fitting my assumption.
```python3
B = (
        (0.8, 0.2),
        (0.5, 0.5),
        (0.1, 0.9),
    )
hmm = DiscreteHMM(len(hidden_var), len(observation_var), B=B)
```
Now, let train the model and see the result:
```python3
hmm.train(obs_seq, verbose=1)
hmm.show_model()
# ----------------------A: Transition probability-----------------------
# [[ 0.33241265  0.60309661  0.06449073]
#  [ 0.46610848  0.21423935  0.31965217]
#  [ 0.29152082  0.10349026  0.60498892]]
# -----------------------B: Emission probability------------------------
# [[ 0.99083784  0.00916216]
#  [ 0.93780741  0.06219259]
#  [ 0.18584415  0.81415585]]
# -------------------pi: initital state distribution--------------------
# [  9.96538870e-01   3.46113023e-03   4.33811166e-25]
```
Appearently, it's an overfitting to this training data.
