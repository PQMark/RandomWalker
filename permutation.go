package main

import (
	"fmt"
	"math/rand"

	randomforest "github.com/malaschitz/randomForest"
)

func permutation(d, test *Dataset, dLabel []int, numIteration, numEstimators, maxDepth, numLeaves int) map[string]float64 {

	var featuresToConsider []string

	// Initialize featuresToConsider to all the features
	featuresToConsider = append(featuresToConsider, d.Features...)

	if len(d.Instance) != len(dLabel) {
		panic("Unequal size of training set and label set")
	}

	//Get the F1 score for the referenceRandomForest
	F1Reference := trainRandomForestPermute(d, test, dLabel, t)

	//map to store average F1 scores for each features
	numFeatures := len(featuresToConsider)
	permutationScores := make(map[string]float64, numFeatures)

	run := 0
	for {
		run++

		// Check the features match with shadows
		CheckFeatures(d.Features, featuresToConsider)

		// Train the RF model numIteration times

		for _, f := range featuresToConsider {
			fPermutationScore := make([]float64, 0, numIteration)

			for i := 0; i < numIteration; i++ {
				fmt.Println("Permute run:", run, "/", i, "/", f)
				current_d := DeepCopy(d, featuresToConsider)
				// Permute features
				current_d.PermuteFeature(f)
				// Train the model and get the decrease of F1 score after shuffeling feature f
				permutedF1Temp := trainRandomForestPermute(d, test, dLabel, featuresToConsider, numEstimators, maxDepth, numLeaves)

				permutationScoreTemp := F1Reference - permutedF1Temp

				fPermutationScore = append(fPermutationScore, permutationScoreTemp)

			}
			//get the average F1 score during numIteration times of permutation of feature f

			permutationScores[f] = Average(fPermutationScore)
		}

		drawPermutationBarplot(permutationScores)

		return permutationScores
	}

}

func (d *Dataset) PermuteFeature(f string) {

	f_shadow := "permute_" + f

	// values stores the values of a feature
	var values []float64

	// Extract the values for the feature
	for _, instance := range d.Instance {
		if val, ok := instance.Features[f_shadow]; ok {
			values = append(values, val)
		}
	}

	// Shuffle the values for shadow
	rand.Shuffle(len(values), func(i, j int) {
		values[i], values[j] = values[j], values[i]
	})

	// Add the shuffled shadow
	for i, instance := range d.Instance {
		instance.Features[f_shadow] = values[i]
	}

}

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
			Class: Y,
		},
		MaxDepth: maxDepth,
		LeafSize: numLeaves,
	}

	forest.Train(numEstimators)

	// Find the threshold of shadow features
	var shadow_IS float64
	featuresNum := len(x[0]) / 2

	for i := featuresNum; i < 2*featuresNum; i++ {
		importanceScore := forest.FeatureImportance[i]

		if importanceScore > shadow_IS {
			shadow_IS = importanceScore
		}
	}

	// Update the results
	// numFeatures := len(x[0])

	for i := 0; i < featuresNum; i++ {
		featureName := d.Features[i]

		if forest.FeatureImportance[i] > shadow_IS {
			results[featureName]++
		} else if _, exists := results[featureName]; !exists {
			results[featureName] = 0
		}

	}

	return F1Score
}

func drawPermutationBarplot(featureScore map[string]float64) {

}
