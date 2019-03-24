# wine_quality_detection

**Introduction:**

The project is a multiclass classification problem to classify quality of wine. 
The data includes red wine and white wine. 

red: {3: 10, 4: 53, 5: 681, 6: 638, 7: 199, 8: 18}   

white: {3: 20, 4: 163, 5: 1475, 6: 2198, 7: 880, 8: 175, 9: 5}

Left column is label and right column is data scale, the larger label means its quality is greater.
The goal is to model wine quality based on physicochemical tests.

The problem is important because it is a multiclass badly imbalanced problem with the smallest class containing only 10 data and the largest class including 681 data. It has 11 features.



**Overview of approaches: **

I used multinomial logistic regression, random forest, adaboost algorithm and xgboost.

1.steps
  -Initially, splitting data into train, validation and test groups with ratio of 6:2:2. 
   -Then recursive feature elimination (RFE) and extra tree classifier were used to select features. 
    -Next, for each method, cross validation was applied to training data to select parameters. 
What should be mentioned is that instead normal accuracy score, I used balanced accuracy score as evaluation rule. I used validation data to choose best selected features among the best 8, 9, 10 and all features, and compared outcome of RFE with that of extra tree classifier. 
     -Finally, compare the outcome of test data to see which model works best.



**Summary of this project: **

For some badly imbalanced problem, especially without large amount of data, it is hard to obtain a high-quality model. In practice, ignoring those minority classes improves model significantly or merging minority class with majority class to make data more balanced. 

One useful method to deal with imbalanced data is that first use One Side Selection (OSS) to under-sample majority classes, then apply Synthetic Minority Over-Sampling Technique (SMOTE) to oversample minority classes. Resampled data outperforms than standardized data slightly (you can see the outcome in project report).
