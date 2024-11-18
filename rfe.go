package main

import (
	"fmt"

	"math"
	"github.com/malaschitz/randomForest"
)

type FeatureStats struct{
	Features []string
	AvgF1 float64
	ErrorF1 float64
}

func REF(d, test *Dataset, dLabel, tLabel []int, numIteration, numEstimators, maxDepth, numLeaves int) []FeatureStats {
	
	results := make([]FeatureStats, 0)

	var featuresToConsider []string		
	// Initialize featuresToConsider to all the features 
	featuresToConsider = append(featuresToConsider, d.Features...)

	//check the size of training data and labels
	if len(d.Instance) != len(dLabel) {
		panic("Unequal size of training set and label set")
	} 

	run := 0 
	for {
		run ++ 

		tempResults := make([]float64, 0, numIteration)
		featureImportances := make([][]float64, numIteration)

		// Train the RF model numIteration times
		for i:=0; i<numIteration; i++ {
			fmt.Println("REF run:", run, "/", i)
			trainRandomForestREF(d, test, dLabel, tLabel, featuresToConsider, numEstimators, maxDepth, numLeaves, &tempResults, &featureImportances[i])
		}

		// Calculate the mean 
		avgF1 := Mean(tempResults)

		// Get the error 
		errorF1 := standardError(tempResults, avgF1)

		featuresToConsiderCopy := make([]string, len(featuresToConsider))
		copy(featuresToConsiderCopy, featuresToConsider)
		stat := FeatureStats{
			Features: featuresToConsiderCopy,
			AvgF1: avgF1,
			ErrorF1: errorF1,
		}

		// Append to result
		results = append(results, stat)

		// Check the number of features remaining 
		if len(featuresToConsider) == 1 {
			return results
		}

		// Discard the features with last 3% FI scores 
		DiscardFeatures(featureImportances, &featuresToConsider, 3)

		fmt.Println(featuresToConsider)

	}
}

// Run one RF 
// Store the F1 score and feature importances
func trainRandomForestREF(d, test *Dataset, dLabel, tLabel []int, features []string, numEstimators, maxDepth, numLeaves int, results *[]float64, featureImportance *[]float64) {

	x := ConvertData(d, features)
	xTest := ConvertData(test, features)

	// Train the RF
	forest := randomforest.Forest{
		Data: randomforest.ForestData{
			X: x,
			Class: dLabel,
		},
		MaxDepth: maxDepth,
		LeafSize: numLeaves,
	}
	forest.Train(numEstimators)

	// Evaluate the trained model on test data 
	predictions := Predict(&forest, xTest)
	F1 := GetF1Score(predictions, tLabel)

	*results = append(*results, F1)

	// Get feature importance 
	importance := make([]float64, 0, len(features))
	for i := 0; i < len(features); i++ {
		importance = append(importance, forest.FeatureImportance[i])
	}

	// Normalize feature importance 
	Normalization(importance)

	// Weigted by F1 score 
	for i := range importance {
		importance[i] *= F1
		*featureImportance = append(*featureImportance, importance[i])
	}

}

func Mean(data []float64) float64{
	sum := 0.0 
	for _, val := range data {
		sum += val 
	}

	return sum / float64(len(data))
}

func standardError(data []float64, mean float64) float64 {
	n := len(data)
	if n == 0 {
		return 0.0
	}

	// Calculate SD 
	sumSquaredDiffs := 0.0
	for _, val := range data {
		diff := val - mean
		sumSquaredDiffs += diff * diff
	}
	std := math.Sqrt(sumSquaredDiffs / float64(n))

	return std / math.Sqrt(float64(n))
}

// Sum to 1
func Normalization(data []float64){
	sum := 0.0 

	for _, val := range data {
		sum += val
	}

	for i := range data {
		data[i] /= sum 
	}
}


// Fine (marginal cases not tested)
func DiscardFeatures(data [][]float64, features *[]string, a int) {
	threshold := float64(a) / float64(100)
	length := len(*features)
	n := float64(len(data))

	size := int(threshold * float64(length))
	if size < 1 {
		size = 1
	}

	importanceMean := make([]float64, length)

	for c := range data[0] {
		sum := 0.0
		for r := range data {
			sum += data[r][c]
		}

		importanceMean[c] = sum / n
	}

	for i := length - 1; i >= 0; i-- { 
		val1 := importanceMean[i]
		count := 0

		for _, val2 := range importanceMean {
			if val1 > val2 {
				count ++
			}

			if count >= size {
				break
			}
		}

		if count < size {
			*features = DeleteFromString((*features)[i], *features)
		}

		if len(*features) <= length - size {
			break
		}

	}

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

/*
func RankFeatureImportances(data [][]float64, features []string) []string {
 
}
*/

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
