# nn

A collection of scripts and notebooks from my exploration of neural-networking. For this study, I'm consulting a collection of whitepapers and some lectures by Andrej Karpathy.

## grad_descent.ipnyb / value.py / nn.py

Basic Autograd-type engine implementing backpropogation over a scalar DAG and minimial neural network library. Inspired by [micrograd](https://github.com/karpathy/micrograd) and intended as a learning instrument.

![nn value tree render](https://raw.githubusercontent.com/danielway/nn/master/images/nn_value_tree_render.png)
![score ground truth](https://raw.githubusercontent.com/danielway/nn/master/images/score_ground_truth.png)
![model score prediction](https://raw.githubusercontent.com/danielway/nn/master/images/score_model_prediction.png)

## bigrams.ipnyb / names.txt

Implementation of a bigram model using PyTorch tensors for next-character prediction. Trained on SSA name data.

![bigram frequencies](https://raw.githubusercontent.com/danielway/nn/master/images/bigram_frequencies.png)

