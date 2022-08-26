# Experts dropout experiment

### Problem description
Sometimes while training some complex models there is a filling
that _you_ now better than your model which input features should
influence more on the model results. 

_P.S. This gut feeling may be wrong, but still..._

But How to tell the model which input features are more valuable
than the others?

Attention? Additional weights? ... 

### Results
TBU

### Solution proposed
**Drop out!** :) 

This repo contains an implementation of the ***Custom Dropout layer*** (CDL)
which can be used as a filter for input features based on the expert's 'importance'
metrics.

For each input feature has to be defined 'importance' metric based on the expert
opinion as a value from 0 to 1, where 

- 1 is **the most valuable feature**
- 0 is **the least valuable feature**.

CDL will be applied as a dropout layer where each input feature is dropped with
the probability `Ti + eps`, where `Ti` is an 'importance' defined.
CDL should be applied to the input features before the rest of the model.

### Dataset
Dataset features description:
- `x1 ... x5, xi in (0...1)`: **x1, x2** are mainly used, **x3** is not the main feature, **x4, x5** are not used.
- `y = 10 * x1 * (x1 + 5 * x2) - 1/100 * x3`
- 10 000 samples

### Experiments configurations
Common configurations: 
- batch_size = 1000
- epochs = 100

Experiments:
1. No custom dropout
2. Regular dropout 
   - 0.1
   - 0.3
   - 0.5
   - 0.9
3. Custom dropout (importance)
   1. Correct:
      - x1 = 1.0, x2 = 1.0, x3 = 1.0, x4 = 0.0, x5 = 0.0
      - x1 = 1.0, x2 = 1.0, x3 = 0.5, x4 = 0.0, x5 = 0.0
      - x1 = 0.8, x2 = 0.8, x3 = 0.3, x4 = 0.05, x5 = 0.05
   2. Incorrect
      - x1 = 0.0, x2 = 0.0, x3 = 0.0, x4 = 1.0, x5 = 1.0
      - x1 = 0.0, x2 = 1.0, x3 = 0.5, x4 = 0.0, x5 = 0.0
      - x1 = 1.0, x2 = 1.0, x3 = 0.0, x4 = 0.0, x5 = 0.0
      - x1 = 0.0, x2 = 1.0, x3 = 0.5, x4 = 1.0, x5 = 0.0
      - x1 = 0.0, x2 = 0.0, x3 = 0.5, x4 = 1.0, x5 = 1.0
   