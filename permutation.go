package main

import (
	"fmt"
	"math/rand"
	"os"
	"os/exec"

	randomforest "github.com/malaschitz/randomForest"
)

func ApplyPermuteMNIST(num int, features []int) {
	d, l := PrepareMnistData(num, features)

	train, label, test, tLabel := SplitTrainTest(d, l, 0.75)

	if err := Write2Json(d, "MNIST.json"); err != nil {
		fmt.Println("Error writing JSON:", err)
	}

	numIteration := 50

	results := permutation(train, test, label, tLabel, numIteration, 30, 10, 10)

	for mode, importanceMap := range results {
		filename := fmt.Sprintf("MNIST_FeatureImportances_%d.json", mode)
		if err := Write2Json(importanceMap, filename); err != nil {
			fmt.Printf("Error writing JSON for mode %d: %v\n", mode, err)
		}
	}

	//get JSON to python
	cmd := exec.Command("python3", "scripts/visualization.py", "MNIST_FeatureImportances_300.json")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Run()
}

// permutationParallel
// Take as input
func permutation(d, test *Dataset, dLabel, tLabel []int, startfeature, numFeatures, numIteration, numEstimators, maxDepth, numLeaves int, F1Reference float64) []FeatureAvgMean {
	var featuresToConsider []string

	// Initialize featuresToConsider to all the features
	featuresToConsider = append(featuresToConsider, d.Features...)

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
	return results
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
