##Variational Auto-encoder

A basic implementation of variational autoencoders using my personal work-flow format that is highly modular and that is GPU-compatible. This repo drew its initial inspirations from [Joost van Amersfoort's repo](https://github.com/y0ast/VAE-Torch). You can run the MNIST experiment by doing:

`./run_gpu`

or

`./run_cpu`


TODO:

1. Add regularization criterion for Gmm prior
2. Add Gaussian mixture with multivariate gaussian component KL divergence criterion
3. Add component weight capability for Gmm
4. Add Logger with visualization functionality for visualizing training