# nn

A collection of scripts and notebooks from my exploration of neural-networking. For this study, I'm consulting a collection of whitepapers and some lectures by Andrej Karpathy.

## grad_descent.ipnyb / value.py / nn.py

Basic Autograd-type engine implementing backpropogation over a scalar DAG and minimial neural network library. Inspired by [micrograd](https://github.com/karpathy/micrograd) and intended as a learning instrument.

_Example NN value tree rendering_

<img src="https://raw.githubusercontent.com/danielway/nn/master/images/nn_value_tree_render.png" width="600" />

_Score ground truth map_

<img src="https://raw.githubusercontent.com/danielway/nn/master/images/score_ground_truth.png" width="300" />

_Model score prediction map_

<img src="https://raw.githubusercontent.com/danielway/nn/master/images/score_model_prediction.png" width="300" />

## bigrams.ipnyb / names.txt

Implementation of a bigram model using PyTorch tensors for next-character prediction. Trained on SSA name data.


_Bigram frequency render_

<img src="https://raw.githubusercontent.com/danielway/nn/master/images/bigram_frequencies.png" width="500" />
