package main

import (
	"fmt"
	"math/rand"
	randomforest "github.com/malaschitz/randomForest"
)

// Fine on toy dataset
// d does not contain label
func permutation(d *Dataset, dLabel []int, numIteration, numEstimators, maxDepth, numLeaves int) ([]string, map[string]int, map[string]float64) {
	removedFeatures := make(map[string]bool) // features marked as unimportant

	var featuresToConsider []string // features remians tentative

	featureImportances := make(map[string]float64)

	// Initialize featuresToConsider to all the features
	for _, name := range d.Features {
		featuresToConsider = append(featuresToConsider, name)
	}

	//check the size of training data and labels
	if len(d.Instance) != len(dLabel) {
		panic("Unequal size of training set and label set")
	}

	run := 0
	for {
		run++

		oldNum := len(featuresToConsider)

		// Make a copy of the data with features to consider
		d := DeepCopy(d, featuresToConsider)

		// Initialize the dataset with shadow features
		d.Initialize(featuresToConsider)

		// Check the features match with shadows
		CheckFeatures(d.Features, featuresToConsider)

		results := make(map[string]int)

		// Train the RF model numIteration times
		for i := 0; i < numIteration; i++ {

			for _, f := range featuresToConsider {
				fmt.Println("Permute run:", run, "/", i, "/", f)
				// Permute features
				d.PermuteFeature(f)
			}

			// Train the model and update the results
			trainRandomForestBoruta(d, dLabel, featuresToConsider, numEstimators, maxDepth, numLeaves, results)
		}

		fmt.Println(results)
		threshold := CalculateThreshold(numIteration)
		fmt.Println("Bionomial Threshold:", threshold)

		// Remove unimportant features
		for f, val := range results {
			if val < threshold {

				// record the feature removed
				removedFeatures[f] = true

				fmt.Println("Delete Feature:", f)

				// remove the feature from featuresToConsider
				featuresToConsider = DeleteFromString(f, featuresToConsider)

			}
		}

		// Converge if there is less than three features or no update any more
		if len(featuresToConsider) < 3 || oldNum == len(featuresToConsider) {
			fmt.Println("Converged.")

			// Train a RF with selected features
			x := ConvertData(d, featuresToConsider)

			forestWithFeatures := randomforest.Forest{
				Data: randomforest.ForestData{
					X:     x,
					Class: dLabel,
				},
			}

			forestWithFeatures.Train(300)

			for i := 0; i < len(featuresToConsider); i++ {
				featureName := d.Features[i]
				featureImportances[featureName] = forestWithFeatures.FeatureImportance[i]
			}

			return featuresToConsider, results, featureImportances
		}

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
