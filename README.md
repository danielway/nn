# nn

A collection of scripts and notebooks from my exploration of neural-networking. For this study, I'm consulting a collection of whitepapers and some lectures by Andrej Karpathy.

## grad_descent.ipnyb / value.py / nn.py

Basic Autograd-type engine implementing backpropogation over a scalar DAG and minimial neural network library. Inspired by [micrograd](https://github.com/karpathy/micrograd) and intended as a learning instrument.

![nn value tree render](https://github.com/danielway/nn/images/nn_value_tree_render.png?raw=true)
![score ground truth](https://github.com/danielway/nn/images/score_ground_truth.png?raw=true)
![model score prediction](https://github.com/danielway/nn/images/score_model_prediction.png?raw=true)

## bigrams.ipnyb / names.txt

Implementation of a bigram model using PyTorch tensors for next-character prediction. Trained on SSA name data.

![bigram frequencies](https://github.com/danielway/nn/images/bigram_frequencies.png?raw=true)

