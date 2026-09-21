# Assignment 8: Variational Autoencoder and Its 2D Latent Space

## Overview

This project trains and compares a **Normal Autoencoder** and a **Variational Autoencoder (VAE)** on the **MNIST handwritten digit dataset**.

Both models use a **2-dimensional latent space**, allowing the encoded representations to be visualized and studied.

### Aim

To train a normal autoencoder and a Variational Autoencoder on MNIST, visualize their 2D latent spaces, compare how they organize handwritten digits, and study how the VAE behaves when decoding points inside and outside the learned latent space.

---

## What This Notebook Covers

1. Loading and displaying the MNIST dataset
2. Building and training a normal autoencoder
3. Visualizing the normal autoencoder's 2D latent space
4. Building and training a Variational Autoencoder (VAE)
5. Using reconstruction loss and KL divergence for VAE training
6. Plotting training and test loss curves
7. Testing image reconstruction
8. Visualizing the VAE latent space
9. Measuring latent-space grouping using 5-nearest-neighbour classification
10. Generating a 15 × 15 decoded latent-space grid
11. Testing extreme latent points
12. Testing a point far outside the learned latent space
13. Comparing the normal autoencoder and VAE
14. Recording observations from the experiment

---

## Dataset

The project uses the **MNIST handwritten digit dataset**.

- Training images: **60,000**
- Test images: **10,000**
- Image size: **28 × 28 pixels**
- Image type: **grayscale**
- Pixel values: scaled to the range **0 to 1**

The notebook uses the MNIST data provided through the PyTorch dataset utilities.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- MNIST dataset
- Adam optimizer
- 5-Nearest-Neighbour classification

---

## Project Structure

```text
Assignment-8/
│
├── Assignment_8_VAE.ipynb
└── README.md
```

> Replace the notebook filename above with the actual filename if it is different in your repository.

---

# 1. Normal Autoencoder

A normal autoencoder consists of:

- **Encoder** – compresses an input image into a small latent representation.
- **Latent space** – contains only two values in this experiment.
- **Decoder** – reconstructs the original image from the latent representation.

Because the latent space has two dimensions, the encoded MNIST images can be displayed on a 2D scatter plot.

The decoder uses a **Sigmoid** output so that reconstructed pixel values remain between 0 and 1.

### Training

The normal autoencoder is trained for:

- **12 epochs**
- Optimizer: **Adam**
- Loss: **Binary Cross Entropy**

The binary cross-entropy loss decreased from **0.2171** in epoch 1 to **0.1735** in epoch 12.

---

# 2. Variational Autoencoder (VAE)

Unlike a normal autoencoder, a VAE does not encode an image as one fixed point.

The encoder produces:

- **Mean (`mu`)**
- **Log variance (`log_var`)**

A latent sample is generated using the reparameterization trick:

```text
z = mu + sigma * epsilon
```

where `epsilon` is random noise sampled from a standard normal distribution.

This makes the sampling process differentiable, allowing the VAE to be trained using backpropagation.

---

## VAE Loss Function

The VAE uses two main loss components.

### Reconstruction Loss

The reconstruction loss measures the difference between the original image and the reconstructed image.

It encourages the decoder to reproduce the input image accurately.

### KL Divergence

The KL divergence measures how far the learned latent distributions are from the standard normal distribution:

```text
N(0, 1)
```

It encourages the latent representations to remain compact and organized around the origin.

### Total Loss

```text
Total Loss = Reconstruction Loss + KL Divergence
```

---

# 3. VAE Training

The VAE is trained for:

- **25 epochs**
- Reconstruction loss: sum of squared pixel differences
- KL divergence regularization
- Training and test losses recorded after every epoch

The total VAE loss per image decreased from **46.063** to **33.929**.

The reconstruction component decreased from **43.107** to **28.667**, while the KL divergence increased from **2.956** to **5.262**.

The final training loss was **33.929**, while the test loss was **34.654**.

The lowest test loss occurred at epoch 24 with **34.559**.

---

# 4. Latent Space Comparison

The experiment compares the 2D latent representations produced by the normal autoencoder and the VAE.

### Normal Autoencoder

The normal autoencoder's latent values spread over a much wider region because there is no explicit constraint forcing the latent representations to remain close to a standard normal distribution.

Observed ranges:

```text
Latent dimension 1: -17.87 to 24.08
Latent dimension 2: -17.78 to 13.45
```

### VAE

The VAE keeps the latent representations much more compact and centered around zero.

Observed ranges:

```text
Latent dimension 1: -3.92 to 3.71
Latent dimension 2: -3.81 to 3.19
```

---

# 5. Digit Clustering

The VAE latent space shows meaningful grouping of similar digits.

Examples observed in the experiment:

- Digit **1** appears near the top.
- Digit **0** appears toward the bottom-left.
- Digit **6** appears near the lower region.
- Digit **7** appears toward the upper-right.
- Digits **4** and **9** have substantial overlap.
- Digits **2, 3, 5, and 8** occupy crowded areas around the middle.

The middle of the latent space is less clearly separated because only two latent dimensions are being used for ten different digit classes.

---

# 6. 5-Nearest-Neighbour Evaluation

A simple **5-nearest-neighbour (5-NN)** classifier is used to quantitatively examine how well digit classes are grouped in the 2D latent space.

The 10,000 test points are divided into two halves:

- One half is used as the reference set.
- The other half is classified according to the majority label of its five closest reference points.

### Results

| Model | 5-NN Accuracy |
|---|---:|
| Normal Autoencoder | 0.773 |
| VAE | 0.799 |

The observed difference is approximately **2.6 percentage points** in this single run.

---

# 7. VAE Reconstruction

The first ten test images are compared with their reconstructed versions.

The reconstructed images are generally recognizable but somewhat blurry.

The mean absolute pixel error for the ten displayed images was:

```text
0.0882
```

Most digits retained their identity. Two of the ten examples were reconstructed as noticeably different digits:

- A **2** appeared more like a **5**
- A **5** appeared more like a **4**

This demonstrates a limitation of representing a complex 28 × 28 image using only two latent dimensions.

---

# 8. Decoded Latent-Space Grid

A **15 × 15 grid** of latent points is created between:

```text
-3 and +3
```

on both latent axes.

Each point is passed through the VAE decoder.

This allows the experiment to visualize how generated digits change as the latent coordinates change.

Observed behavior includes transitions such as:

```text
0 → 6 → 4
8 → 5 → 3 → 2 → 4 → 9 → 7
```

The exact generated shapes vary across the grid, but the overall transitions correspond to regions occupied by different digit classes.

---

# 9. Smoothness of the Latent Space

Inside the region where most training data is located, neighboring latent points generally produce gradual changes in the decoded images.

Examples include:

- Strokes gradually tilting
- Loops opening or closing
- One digit gradually changing into another
- The digit 0 becoming narrower and changing toward a 6

Some areas, particularly near the center, become blurry because several digit classes compete for the same small region of the 2D latent space.

---

# 10. Extreme Latent Points

The experiment also tests points near and beyond the main region occupied by the encoded test data.

The VAE still produces recognizable shapes at some moderately extreme points.

Examples observed:

- `(-5, -5)` → clear 0-like output
- `(0, -5)` → clear 0-like output
- `(-5, 5)` → thin 7-like output
- `(-3.5, 3.5)` → thin 7-like output
- `(0, 5)` → thin 1-like stroke
- `(5, 5)` → broken fragments
- `(-5, 0)` → overlapping 8/3-like mixture
- `(5, 0)` → thick, blocky 7-like shape

---

# 11. Point Far Outside the Latent Space

The experiment tests the point:

```text
(-12, 12)
```

This point is far outside the region where the VAE learned most of its latent representations.

The decoder produces a sharp, high-contrast 7-like pattern rather than a natural handwritten digit.

The observed pixel statistics were:

```text
Minimum pixel value: 0.0000
Maximum pixel value: 1.0000
Mean pixel value:    0.0350
```

When moving farther along the same direction, the output changes rapidly at first and then becomes increasingly stable.

Mean pixel values recorded at distances 0, 2, 4, 6, 8, 12 and 20 were:

```text
0.1540
0.0931
0.0449
0.0393
0.0371
0.0353
0.0347
```

This indicates that the decoder produces a saturated pattern when it receives latent coordinates far outside the region seen during training.

---

# 12. Normal Autoencoder vs VAE

| Property | Normal Autoencoder | VAE |
|---|---:|---:|
| Latent range, dimension 1 | -17.87 to 24.08 | -3.92 to 3.71 |
| Latent range, dimension 2 | -17.78 to 13.45 | -3.81 to 3.19 |
| Average distance between digit centres | 6.35 | 1.67 |
| 5-NN accuracy | 0.773 | 0.799 |

The normal autoencoder produces a much wider latent representation.

The VAE keeps its latent representations compact and centered around zero because of the KL divergence regularization.

This organized latent space allows the VAE to generate images by sampling points around the learned distribution.

---

# 13. Training Behaviour

### Normal Autoencoder

The binary cross-entropy loss decreased from:

```text
Epoch 1:  0.2171
Epoch 12: 0.1735
```

The largest reduction occurred during the first few epochs.

### VAE

The total loss decreased from:

```text
46.063 → 33.929
```

The reconstruction loss decreased from:

```text
43.107 → 28.667
```

The KL divergence increased from:

```text
2.956 → 5.262
```

The increase in KL divergence occurs as the encoder learns to organize different digits in the latent space while reconstruction quality improves.

The training and test losses remained relatively close, with final values of:

```text
Training: 33.929
Test:     34.654
```

---

# 14. Key Findings

1. The VAE produces a more compact 2D latent representation.
2. Similar handwritten digits tend to form nearby groups.
3. Some digit classes are clearly separated, while others overlap.
4. The 2D restriction makes complete separation of all ten classes difficult.
5. The VAE latent space supports smooth transitions between generated digits.
6. The KL divergence encourages latent representations to stay around the standard normal distribution.
7. Reconstructions are recognizable but can be blurry.
8. Points moderately outside the learned latent region can still produce digit-like images.
9. Very distant latent points can produce unnatural or saturated patterns.
10. In this run, the VAE achieved a 5-NN accuracy of **0.799**, compared with **0.773** for the normal autoencoder.

---

# Conclusion

This experiment demonstrates how a **Variational Autoencoder** can organize MNIST images into a structured 2D latent space.

Compared with the normal autoencoder, the VAE keeps encoded representations more compact and centered around the origin. Similar digits tend to occupy nearby regions, and moving through the latent space generally produces gradual changes in generated images.

At the same time, using only two latent dimensions introduces limitations. Several digit classes overlap, reconstructions can become blurry, and points far outside the learned latent region can produce unnatural outputs.

Overall, the notebook provides a practical demonstration of:

- Autoencoders
- Variational Autoencoders
- Latent-space representation
- Reparameterization
- Reconstruction loss
- KL divergence
- MNIST image reconstruction
- Latent-space visualization
- Generative sampling

---

## How to Run

### 1. Install the required libraries

```bash
pip install torch torchvision numpy matplotlib
```

### 2. Open the notebook

The project can be run using:

- Jupyter Notebook
- JupyterLab
- Google Colab
- VS Code with Jupyter support

### 3. Run all cells

Run the notebook from the first cell to the last cell so that:

1. MNIST is downloaded/loaded.
2. The normal autoencoder is trained.
3. The VAE is trained.
4. Loss curves are generated.
5. Reconstructions are displayed.
6. Latent spaces are plotted.
7. The decoded latent grid is generated.
8. Extreme latent-point experiments are performed.
9. Written observations are displayed.

---

## Reproducibility

The notebook uses a fixed random seed:

```text
7
```

The numerical observations documented in this README correspond to the notebook run using that seed.

---

## Author

**Nitheesh**

BCA Student  
AI/ML Enthusiast
