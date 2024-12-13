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

To use the Boruta feature selection method:
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
Sample output with MNIST database, representing selection of best 50/784 features. 

![Boruta MNIST visualization](results_images/Boruta_MNIST.png)

# RFECV

Recursive feature elimination with cross validation aims to iteratively remove a threshold of features with the lowest feature importance scores until a single feature remains. Threshold is determined by the power law of decay until the 20% feature remains:
$$
\text{threshold} = \text{initial threshold} \times \% \, \text{remaining features} ^ \text{decay factor}
$$

To use RFECV:

- For each fold across `numFolds` use the `trainRandomForestRFE()` method to repeatedly train the random forest model `numIteration` times, keeping track of feature importances
- Use `FeatureDecayScheduler()` to determine a threshold and eliminate features with `DiscardFeatures()`

```
for i := range featuresToConsider {
	for j := 0; j < numIteration; j++ {
		avgFeatureImportance[i] += featureImportances[j][i]
	}
	avgFeatureImportance[i] /= float64(numIteration)
	importanceScores[featuresToConsider[i]] = avgFeatureImportance[i]
}
    ...
    // using power law decay
    threshold := FeatureDecayScheduler(&featuresToConsiderCopy, len(d.Features), lrParams)

    // Discard the features
    DiscardFeatures(featureImportances, &featuresToConsider, threshold)         

```
Sample output with MNIST with the best 50 features selected by RFECV:


# Permutation
Here is an example of `inline code` formatting in Markdown.

# mRMR

# Acknowledgements


