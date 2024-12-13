# A Feature Selection Bundle

Feature selection is the task of identifying a subset of the most relevant features from a feature space to improve model training in machine learning. By eliminating non-consistent, redundant and irrelevant features, we reduce the number of input variables, retaining only the features that show the strongest relationship to the target variable. Here, we demonstrate 4 implementation of feature selection methods using the Random Forest learning model. To ensure stability of feature selection, user can specify the number of folds (`numFolds`) for k-fold cross-validation. 
- Boruta
- Recursive Feature Elimination 
- Permutation
- mRMR 

# Random Forest

Random forest is implemented with hyperparameters such as the number of decision trees (`NTrees`), depth of the tree (`MaxDepth`), and size of the leaf (`LeafSize`) for training the model. 


# Boruta

The Boruta algorithm performs by iteratively identifying and removing features with importance scores lower than than of noise features, effectively retaining features of the highest relevance. 

To use the Boruta feature selection method
- Use the `GridSearchParallel()` method to identify the best set of hyperparameters or proceed with the default parameters
- To select features, run the `Boruta()` function to train the random forest model `numIteration` times, using the `ShuffleShadowFeatures()` method and `trainRandomForestBoruta()` method to shuffle shadow features, train the model 
and update results over each iteration
    ```
    for i := 0; i < numIteration; i++ {

		// Shuffle Shadow features
		d.ShuffleShadowFeatures(featuresToConsider)

		// Train the model and update the results
		trainRandomForestBoruta(d, dLabel, featuresToConsider, numEstimators, maxDepth, numLeaves, results)
    }
    ```

- Train RF with selected features and tunned hyperparameters, then use the `Predict()` method to evaluate model performance on outer test dataset

```
	// Evaluate the model on outer test
	outerTestProcessed := ConvertData(outerTest, outerTest.Features)
	predictions := Predict(&forest, outerTestProcessed)
	f1 := GetF1Score(predictions, outerLabel)
```
Sample output with MNIST database, representing selection of 50/784 features. 

![Boruta MNIST visualization](results_images/Boruta_MNIST.png)

# RFE


# Permutation
Here is an example of `inline code` formatting in Markdown.

# mRMR

# Acknowledgements


