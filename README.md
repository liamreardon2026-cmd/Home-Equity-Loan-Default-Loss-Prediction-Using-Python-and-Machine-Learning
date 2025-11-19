For this project in my MSDS 422 Practical Machine Learning course, I was provided an uncleaned dataset containing information on home-equity line-of-credit loans, including borrower default status and associated loss amounts. The dataset also included predictive features such as employment details, income levels, credit history, and additional borrower characteristics.

I cleaned and prepared the dataset by correcting outliers, resolving missing values, and standardizing the data before modeling. Using Python, I developed a series of predictive models to classify default risk and estimate loss amounts. The models I implemented included:
• Decision Trees
• Random Forests
• Gradient Boosting
• Regression with all variables
• Regression with feature sets selected from each model (Decision Trees, Random Forests, Gradient Boosting)
•Stepwise Regression
•TensorFlow Neural Network

Based on model performance, I concluded that the random forest model was the most effective for predicting borrower default, achieving 92% accuracy on the testing set. For predicting loss amount, gradient boosting provided the strongest results with a 2,556 RMSE. While I typically prefer simpler models when performance is comparable, these approaches demonstrated clear superiority over the alternatives. 

The uncleaned dataset and Python code will be included in my GitHub repository linked below. I have also added a supporting document with the ROC curve, classification accuracy, and loss accuracy. Feel free to reach out if you would like to explore the dataset or test the model yourself.
