# Brainiac Model 

The Brainiac model is based on https://www.sciencedirect.com/science/article/pii/S1878929325000647 where it is used for the analysis of fMRI data.

From a computational perspective it stands out as the projection matrix $A$ mapping the latent variables to the observations is dense, resulting in a dense conditional precision matrix $Q_c$ (while $Q_p$ is diagonal). It also has the particularity that the variance in the observations is coupled to a hyperparameter from the model. More concretely we have that 

$$
Q_p([h, \alpha_1, ..., \alpha_d]) = h^2 \ \Phi(\alpha)
$$
while $y \sim N(0, 1-h^2 I)$. 
Thus, while fitting the LGM framework, it doesn't cohere with the usual R-INLA setup.
More details can be found in `generate_data.py` where the dataset (and thus the model) is generated. 

## Scripts

- **`generate_data.py`**: Generates a synthetic dataset given the dimensions specified at the top of the file. 

- **`run.py`**: Loads the generated dataset (requires the correct dimension to be specified at the top of the file) and runs DALIA. 

The GPU-backend is especially suitable for this model. 

## Usage

```bash
export ARRAY_MODULE=cupy
python generate_data.py
python run.py
```