package main

import (
	"fmt"
	"math/rand"

	randomforest "github.com/malaschitz/randomForest"
)

// Nice Work !!!!! 
// 1. Run RF several times to get a more stable F1 score baseline
// 2. Incorporate standard error (use the standardError and Average functions)
// 3. Optimize for parallel execution
// 4. Write a script for visualization

func Permutation(d, test *Dataset, dLabel, tLabel []int, numIteration, numEstimators, maxDepth, numLeaves int) map[string]float64 {

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
	permutationScores := make(map[string]float64, numFeatures)

	// Train the RF model numIteration times

	for _, f := range featuresToConsider {
		fPermutationScore := make([]float64, 0, numIteration)
		current_d := DeepCopy(d, featuresToConsider)

		for i := 0; i < numIteration; i++ {
			fmt.Println("Permutation run:", i, "/", f)

			// Permute features
			current_d.PermuteFeature(f)
			// Train the model and get the decrease of F1 score after shuffeling feature f
			permutedF1Temp := trainRandomForestPermute(current_d, test, dLabel, tLabel, featuresToConsider, numEstimators, maxDepth, numLeaves)

			permutationScoreTemp := F1Reference - permutedF1Temp

			fPermutationScore = append(fPermutationScore, permutationScoreTemp)

		}
		//get the average F1 score during numIteration times of permutation of feature f

		permutationScores[f] = Average(fPermutationScore)
	}

	return permutationScores

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
