# Bias-Variance Trade-Off Simulation

This Python module is dedicated to understanding the fundamental trade-off between bias and variance in statistical learning models, particularly in the context of linear regression. It provides a simulation framework to explore how different factors affect the accuracy and reliability of regression models.

## Principles

The bias-variance trade-off is a key concept in machine learning and statistics. It refers to the tension between a model's ability to minimize errors on the training data (bias) and its ability to generalize well to new, unseen data (variance).

- **Bias**: The error introduced by a model's simplifications or assumptions. High bias can lead to underfitting, where the model fails to capture the underlying patterns in the data.
- **Variance**: The variability in the model's predictions for different training datasets. High variance can lead to overfitting, where the model closely fits the training data but does not perform well on new data.

Finding the right balance between bias and variance is essential for building models that are both accurate and robust.

## Simulation Process

The simulation involves generating synthetic data with a known underlying trend, then adding noise to simulate real-world data. Linear regression models of increasing complexity are fit to this data, and their performance is evaluated based on their ability to predict the true underlying trend.

Real function $y=2{e}^x+\epsilon$ , in it $\epsilon\sim N(0,\sigma^2)$ . Taylor expansion of $f(x) = 2{e}^x$ , $$f(x) = 2{e}^x =2 +  2x + x^2 + x^3/3 + ...$$

So, different complexity functions $h(x)$ were chosen for training, and the complexity of the following models increased from low to high:

 $$h_1(x) = a_1x+b$$

 $$h_2(x) = a_1x+a_2x^2 + b$$

 $$h_3(x) = a_1x+a_2x^2+a_3x^3+b$$

## Outcomes

The simulation provides insights into the following metrics for each model complexity level:

- Mean squared bias: A measure of how well the model captures the true trend.
- Variance: A measure of how much the model's predictions fluctuate with different training data.
- Noise variance: The inherent variability in the data.
- Total error: The combination of bias and variance, which represents the overall prediction error.

## Dependencies

- `numpy`: For numerical computations.
- `pandas`: For data manipulation and presentation.
- `scipy`: For advanced numerical routines.
