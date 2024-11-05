package main

import (
	"fmt"

	"github.com/malaschitz/randomForest"
)

// returns the list of important feature after performing rfe using random forest model
func RecursiveFeatureElimination(d *Dataset, dLabel []int, numEstimators int, threshold float64, targetFeatureNumber int) []string {
	// range over all features and append to remaining features
	var remainingFeatures []string
	for _, name := range d.Features {
		remainingFeatures = append(remainingFeatures, name)
	}

    // map of feature to importance
	results := make(map[string]float64)

	// continues until target feature number reached
	for len(remainingFeatures) > targetFeatureNumber {
        results = make(map[string]float64)
        
		feature := trainRF(d, dLabel, remainingFeatures, numEstimators, results)

		remainingFeatures = DeleteFromString(feature, remainingFeatures)
		fmt.Printf("Remaining features: %v\n", remainingFeatures)
	}

	return remainingFeatures

}

// returns the least important feature
func trainRF(d *Dataset, Y []int, features []string, numEstimators int, results map[string]float64) string {
	// Convert dataset to the required format
	x := ConvertData(d, features)

	forest := randomforest.Forest{
		Data: randomforest.ForestData{
			X:     x,
			Class: Y,
		},
	}

	// Train the random forest
	forest.Train(numEstimators)

	// Determine the least important feature based on feature importances
	minIndex := getLeastImportantFeatureIndex(forest.FeatureImportance)

	// record importance for each feature in results map
	results[features[minIndex]] = forest.FeatureImportance[minIndex]

	return features[minIndex]
}

// takes in a Dataset object and a slice of features, returns data in the format of [][]float64
func ConvertData(d *Dataset, features []string) [][]float64 {

	data := make([][]float64, len(d.Instance))

	// ranging over all the datapoints of d
	for i, instance := range d.Instance {
		data[i] = make([]float64, 0, len(features))

		// for each row, append all value for each feature
		for _, f := range features {
			data[i] = append(data[i], instance.Features[f])
		}
	}

	return data
}

// getLeastImportantFeatureIndex returns the index of the least important feature from the given importance slice.
func getLeastImportantFeatureIndex(importances []float64) int {
	min := importances[0]
	minIndex := 0
	for i, num := range importances {
		if num < min {
			min = num
			minIndex = i
		}
	}
	return minIndex
}

// RFE performs Recursive Feature Elimination using a Random Forest model. (golearn)
// func RFE(trainData, testData *base.DenseInstances, numEstimators int, threshold float64) ([]string, error) {
// 	// Get initial set of features
// 	// get a slice of all attributes
// 	remainingFeatures := trainData.AllAttributes()
// 	featureNames := make([]string, len(remainingFeatures))

// 	// range over the slice of attributes and append to feature names
// 	for i, attribute := range remainingFeatures {
// 		featureNames[i] = attribute.GetName()
// 	}

// 	featureCount := int(math.Sqrt(float64(len(remainingFeatures))))

// 	for len(remainingFeatures) > 1 {
// 		// Train a new Random Forest model with numEstimator number of trees and featureCount
// 		// number of features to build each tree
// 		rf := ensemble.NewRandomForest(numEstimators, featureCount)
// 		if err := rf.Fit(trainData); err != nil {
// 			return nil, fmt.Errorf("failed to fit model: %s", err)
// 		}

// 		// Get feature importance
// 		importances := rf.FeatureImportance() //not a method
// 		if len(importances) != len(remainingFeatures) {
// 			return nil, fmt.Errorf("feature importance size mismatch")
// 		}

// 		// Find feature with minimum importance
// 		minImportance := importances[0]
// 		minIndex := 0
// 		for i, importance := range importances {
// 			if importance < minImportance {
// 				minImportance = importance
// 				minIndex = i
// 			}
// 		}

// 		// Check threshold condition
// 		if minImportance >= threshold {
// 			break
// 		}

// 		// Remove feature from remainingFeatures and featureNames
// 		remainingFeatures = append(remainingFeatures[:minIndex], remainingFeatures[minIndex+1:]...)
// 		featureNames = append(featureNames[:minIndex], featureNames[minIndex+1:]...)

// 		// Update training and testing datasets by dropping the column
// 		// trainData, _ = base.DropColumn(trainData, trainData.AllAttributes()[minIndex])
// 		// testData, _ = base.DropColumn(testData, testData.AllAttributes()[minIndex])
// 	}

// 	return featureNames, nil
// }
