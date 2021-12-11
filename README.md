# Denoising-AutoEncoder
The Denoising Autoencoder is an extension of the autoencoder. Just as a standard autoencoder, it’s composed of an encoder, that compresses the data into the latent code, extracting the most relevant features, and a decoder, which decompress it and reconstructs the original input. There is only a slight modification: the Denoising Autoencoder takes a noisy image as input and the target for the output layer is the original input without noise.

[Reference](https://ai.plainenglish.io/denoising-autoencoder-in-pytorch-on-mnist-dataset-a76b8824e57e)