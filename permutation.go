package main

import (
	"fmt"
	"math/rand"
	"runtime"

	randomforest "github.com/malaschitz/randomForest"
)

func permutation(d, test *Dataset, dLabel, tLabel []int, numIteration, numEstimators, maxDepth, numLeaves int) []FeatureAvgMean {

	var featuresToConsider []string

	// Initialize featuresToConsider to all the features
	featuresToConsider = append(featuresToConsider, d.Features...)

	if len(d.Instance) != len(dLabel) {
		panic("Unequal size of training set and label set")
	}

	//Get the F1 score for the referenceRandomForest
	F1Reference := trainRandomForestPermute(d, test, dLabel, tLabel, featuresToConsider, numEstimators, maxDepth, numLeaves)

	//map to store average F1 scores for each features
	numFeatures := len(featuresToConsider)
	results := make([]FeatureAvgMean, 0, numFeatures)

	numCPU := runtime.NumCPU()
	featuresEachCPU := numFeatures / numCPU
	c := make(chan []FeatureAvgMean, numCPU)

	for i := 0; i < numCPU; i++ {
		if i != numCPU-1 {
			startfeature := i * featuresEachCPU
			go permutationFeaturesParallel(d, test, dLabel, tLabel, featuresToConsider, startfeature, featuresEachCPU, numIteration, numEstimators, maxDepth, numLeaves, F1Reference, c)
		} else {
			numFeaturesLastCPU := numFeatures - featuresEachCPU*(numCPU-1)
			startfeature := i * featuresEachCPU
			go permutationFeaturesParallel(d, test, dLabel, tLabel, featuresToConsider, startfeature, numFeaturesLastCPU, numIteration, numEstimators, maxDepth, numLeaves, F1Reference, c)
		}
	}

	for i := 0; i < numCPU; i++ {
		resultTemp := <-c
		results = append(results, resultTemp...)
	}

	return results

}

// permutationParallel
// Take as input
func permutationFeaturesParallel(d, test *Dataset, dLabel, tLabel []int, featuresToConsider []string, startfeature, numFeatures, numIteration, numEstimators, maxDepth, numLeaves int, F1Reference float64, c chan []FeatureAvgMean) {
	results := make([]FeatureAvgMean, numFeatures)
	for j := startfeature; j < startfeature+numFeatures; j++ {
		F1Score := make([]float64, 0, numIteration)

		for i := 0; i < numIteration; i++ {
			fmt.Println("Permute run:", i, "/", featuresToConsider[j])
			current_d := DeepCopy(d, featuresToConsider)
			// Permute features
			current_d.PermuteFeature(featuresToConsider[j])
			// Train the model and get the decrease of F1 score after shuffeling feature f
			permutedF1Temp := trainRandomForestPermute(d, test, dLabel, tLabel, featuresToConsider, numEstimators, maxDepth, numLeaves)

			F1Score = append(F1Score, permutedF1Temp)

		}
		//get the average F1 score during numIteration times of permutation of feature f

		// Calculate the mean
		avgF1 := Average(F1Score)
		avgPermut := avgF1 - F1Reference
		// Get the error
		errorPermut := standardError(F1Score, avgF1)

		results[j-startfeature] = FeatureAvgMean{
			Feature:          featuresToConsider[j],
			AvgPermutScore:   avgPermut,
			ErrorPermutScore: errorPermut,
		}
	}
	c <- results
}

// function on object d
// take as input string f, which is the name of the feature that will be permuted
// The function shufffle the value of given feature amoung all samples
// no out put
func (d *Dataset) PermuteFeature(f string) {

	// values stores the values of a feature
	var values []float64

	// Extract the values for the feature
	for _, instance := range d.Instance {
		if val, ok := instance.Features[f]; ok {
			values = append(values, val)
		}
	}

	// Shuffle the values for shadow
	rand.Shuffle(len(values), func(i, j int) {
		values[i], values[j] = values[j], values[i]
	})

	// Add the shuffled shadow
	for i, instance := range d.Instance {
		instance.Features[f] = values[i]
	}

}

// Take as input two datasets d and test, and the corresponding labels dLabel and tLabel as slice of integers, a slice of string features represent all features, and three integers as parameaters for random forest
// Output: A float64 number, The F1 score after running randomforest
func trainRandomForestPermute(d, test *Dataset, dLabel, tLabel []int, features []string, numEstimators, maxDepth, numLeaves int) float64 {
	var F1Score float64
	// Prepare training data and labels for training process
	// convert the data to [][]float64 type
	x := ConvertData(d, features)
	xTest := ConvertData(test, features)
	//trainY is dLabel

	forest := randomforest.Forest{
		Data: randomforest.ForestData{
			X:     x,
			Class: dLabel,
		},
		MaxDepth: maxDepth,
		LeafSize: numLeaves,
	}

	forest.Train(numEstimators)

	//Estimate model persision get F1 score
	predictions := Predict(&forest, xTest)
	F1Score = GetF1Score(predictions, tLabel)

	return F1Score
}

//End
