Experimenting with machine learning for self learning!

# Linear Regression

Linear regression is very simple: it finds the line of best fit for a given set of $(x, y)$ coordinate pairs. That means finding the best parameters $m$ (slope) and $b$ (y-intercept). We define the cost function as:

$$C(x,y) = \frac{1}{n} \sum^n_{i=1}((mx_i+b) - y_i)^2$$

which is simply MSE (Mean Squared Error). Therefore, the gradient is given by the derivates of the cost function in terms of $m$ and $b$, which gives:

$$\frac{\partial C}{\partial m} = \frac{2}{n} \sum^n_{i=1} x_i ((mx_i+b) - y_i)$$
$$\frac{\partial C}{\partial b} = \frac{2}{n} \sum^n_{i=1} ((mx_i+b) - y_i)$$

We can then take a small step based on the learning rate to adjust the parameters against the direction of the derivatives.

In `linearregressionfromscratch.py` I implement linear regression without copying code from online resources just for fun. I use matplotlib's `FuncAnimation` to show how the line evolves over time! Convergence is pretty quick, but a larger learning rate results in exploding gradients, because then the update step overshoots the minimum, and the error grows instead of shrinking, which compounds on the next step. Normalizing the input vector $x$ might solve this as well as just using a smaller learning rate.
