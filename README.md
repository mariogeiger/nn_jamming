# Jamming in neural networks

Code used in https://arxiv.org/abs/1809.09349

Depends on https://github.com/mariogeiger/hessian

### Fully-connected network trained with a quadratic hinge loss

The main file is `train.py`

Usage example:
```bash
python train.py --log_dir R40d40h5L --dataset random --dim 40 --width 40 --depth 5 --p 24000 --optimizer adam --init orth --n_steps_max 1M --learning_rate 0.1 --max_learning_rate 0.1
```

It will train a neural network of 5 hidden layers of 40 units on a random dataset in dimension 40.
It will create a directory called `R40d40h5L` and save the logs and and many measures of the run.

The script has many parameters, [see here](https://github.com/mariogeiger/nn_jamming/blob/master/train.py#L18-L50).

Here is an example of code that load the results of a run
```python
from functions import *  # functions.py contains useful functions

log_dir = "R40d40h5L"
runs = list(load_dir2(log_dir))
print("{} runs in this directory".format(len(runs))  # 1 if you ran the command above

run = runs[0]  # pick the first one
print("{depth} layers of {width} units trained on {p} points".format(**run['desc']))

dynamics = run['dynamics']  # list containing many measures during the training
print("The finall loss is {}".format(dynamics[-1]['train'][1]))

model_init, model_last, trainset, testset = load_run(runs[0])
```
