Machine Learning (CS60050): Assignment 1
Download data from
https://drive.google.com/open?id=1vJwvlt8Tp-mwbDj1V0ePOe0Ej1ndWXOx
In the first four columns of each row, you have four features corresponding to a house -
area in square feet, number of floors, number of bedrooms and number of bathrooms. In the
fifth column of each row, you will find the price of the house. The task is to predict the price
of the houses from their corresponding features using linear regression.
For all the questions below, use the first 80% of the rows for training and the remaining
20% for testing, i.e., measuring the performance of the learned model. ​The performance
of the learned model is measured using Root Mean Square Error (RMSE) which is defined
as,
where Zfi
is the predicted price and Zoi
is the actual price of the i-th house in the test set (size
N). Note that the test set (last 20% of the rows) should NOT be used to train the model. Also
the prices (last column) in the test set should not be visible to the model, rather these values
should be used only for measuring the performance.
Part (a): implementing linear regression
- Use linear combination of the features.
- Minimize mean squared error cost function (as discussed in class).
- Use gradient descent to minimize the cost function (as discussed in class). Use
learning rate of 0.05.
- Solve the problem with and without regularization. Show how the test RMSE varies
with the weightage of the regularization terms (use same weightage for all features).
Part (b): experimenting with optimization algorithms
- Use linear combination of the features.
- Minimize mean squared error cost function (as discussed in class). Do not use any
regularization.
- Solve the problem by minimizing the cost function using two optimization algorithms:
(i) gradient descent with learning rate of 0.05, and (ii) iterative re-weighted least
square method (described at
http://www.cedar.buffalo.edu/~srihari/CSE574/Chap4/4.3.3-IRLS.pdf).
- Plot the test RMSE vs number of iterations for both the optimization algorithms.
Which optimization algorithm would you prefer for this problem and why?
Part (c): experimenting with combinations of features
- Minimize mean squared error cost function. Do not use any regularization.
- Use gradient descent to minimize the cost function. Use learning rate of 0.05.
- Solve the problem using (i) linear, (ii) quadratic and (iii) cubic combinations of the
features.
- Plot the test RMSE vs learning rate for each of the cases. Which one you would
prefer for this problem and why?
Part (d): experimenting with cost functions
- Use linear combination of the features.
- Solve the problem by minimizing different cost functions: (i) mean absolute error, (ii)
mean squared error, and (iii) mean cubed error. Do not use any regularization.
- Use gradient descent to minimize the cost function in each case.
- Plot the test RMSE vs learning rate for each of the cost functions. Which one would
you prefer for this problem and why?
