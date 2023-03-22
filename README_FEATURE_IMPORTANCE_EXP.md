# Experts dropout experiment - permutation feature importance

## Experiments to run
1. overfit model, train model
2. add simple dropout, repeat training
3. remove some parameters using permutation importance
4. use dropout based on feature importance
   - all 
   - the most valuable
   - several 

## The second experiment - kaggle Titanic
...

## The first experiment - flatten MNIST
I guess, MNIST is not the best choice for testing this experiment, because
it has 784 parameters so it's not easy to calculate permutation feature
importance on it. And besides that all 784 parameters are almost equally
important for predicting label. So it's been decided to run experiment
on the other dataset.

*P.S. The experiment may actually improve some results limiting the model
to overfit on some specific pixels, but it would require some extra
computational power to test*


### TODO
1. implement feature importance as a separate script which dumps the results
2. load feature importance results during model configuration
3. load feature importance results during dataset filtering
4. run all the rest experiments
5. update readme, add scrinshots, etc.
6. test tensorboard hub 
7. apply streamlit? (before - after showcase on mnist)
8. merge current branch into master
