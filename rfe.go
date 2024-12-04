package main

import (
	"fmt"

	randomforest "github.com/malaschitz/randomForest"
)

type FeatureStats struct {
	Features []string
	AvgF1    float64
	ErrorF1  float64
}

// outputs optimal number of features 
func RFECV(data *Dataset, labels []int, numIteration, numEstimators, maxDepth, numLeaves, numFolds, numFeatures int) int {
	dataFolds, labelFolds := FoldSplit(data, labels, numFolds)
	results := make([][]float64, numFolds)

	for i := 0; i < numFolds; i++ {
		innerTrain, innerLabel, outerTest, outerLabel := GetFoldData(dataFolds, labelFolds, i)

		featureStats, _ := RFE(innerTrain, outerTest, innerLabel, outerLabel, numIteration, numEstimators, maxDepth, numLeaves, 0)

		// range over all the feature stats in list feature stat
		// set each col of result[i] as f1 score
		for j := range results[i] {
			if j < len(featureStats) {
				results[i][j] = featureStats[j].AvgF1
			}
		}

	}

	// map to store feature count to avg score
	avgScores := make(map[int]float64, 0)

	// range over all the folds for each feature count
	for j := range results[0] {
		total := 0.0

		for i := range numFolds {
			total += results[i][j]			
		}

		avgScores[j+1] = total/float64(numFolds)
	}

	// range over the map to identiy feature count with the highest f1 score
	maxAvgF1 := 0.0
	optimalFeatureCount := 0
	for featureCount, AvgF1 := range avgScores {
		if AvgF1 > maxAvgF1 {
			maxAvgF1 = AvgF1
			optimalFeatureCount = featureCount
		}
	}


	return optimalFeatureCount

}

// numFeatures sets the minimun number of features to stop rfe, 0 means run until single feature
// returns list of FeatureStats and importance scores
func RFE(d, test *Dataset, dLabel, tLabel []int, numIteration, numEstimators, maxDepth, numLeaves int, numFeatures int) ([]FeatureStats, map[string]float64) {

	results := make([]FeatureStats, 0)
	thresholdFeatures := false

	if numFeatures != 0 {
		thresholdFeatures = true
	}


	var featuresToConsider []string
	// Initialize featuresToConsider to all the features
	featuresToConsider = append(featuresToConsider, d.Features...) 

	//check the size of training data and labels
	if len(d.Instance) != len(dLabel) {
		panic("Unequal size of training set and label set")
	}

	run := 0
	for {
		run++

		tempResults := make([]float64, 0, numIteration)
		featureImportances := make([][]float64, numIteration)

		// Train the RF model numIteration times
		for i := 0; i < numIteration; i++ {
			fmt.Println("REF run:", run, "/", i)
			trainRandomForestRFE(d, test, dLabel, tLabel, featuresToConsider, numEstimators, maxDepth, numLeaves, &tempResults, &featureImportances[i])
		}
		
		importanceScores := make(map[string]float64)
		avgFeatureImportance := make([]float64, len(featuresToConsider))
		for i := range featuresToConsider {
			for j := 0; j < numIteration; j++ {
				avgFeatureImportance[i] += featureImportances[j][i]
			}
			avgFeatureImportance[i] /= float64(numIteration)
			importanceScores[featuresToConsider[i]] = avgFeatureImportance[i]
		}

		// Calculate the mean
		avgF1 := Average(tempResults)

		// Get the error
		errorF1 := standardError(tempResults, avgF1)

		featuresToConsiderCopy := make([]string, len(featuresToConsider))
		copy(featuresToConsiderCopy, featuresToConsider)
		stat := FeatureStats{
			Features: featuresToConsiderCopy,
			AvgF1:    avgF1,
			ErrorF1:  errorF1,
		}

		// Append to result
		results = append(results, stat)

		// if min number of features indicated, return when len(features) == numFeatures
		// else repeat until single feature 
		if thresholdFeatures {
			if len(featuresToConsider) <= numFeatures {
				singleResult := make([]FeatureStats, 0)
				singleResult = append(singleResult, results[run-1])
				return singleResult, importanceScores

			}

		} else {
			if len(featuresToConsider) == 1 {
				return results, importanceScores
			}
		}


		// Discard the features with last 3% FI scores
		DiscardFeatures(featureImportances, &featuresToConsider, 3)

		fmt.Println(featuresToConsider)

	}
}

// Run one RF
// Store the F1 score and feature importances
func trainRandomForestRFE(d, test *Dataset, dLabel, tLabel []int, features []string, numEstimators, maxDepth, numLeaves int, results *[]float64, featureImportance *[]float64) {

	x := ConvertData(d, features)
	xTest := ConvertData(test, features)

	// Train the RF
	forest := randomforest.Forest{
		Data: randomforest.ForestData{
			X:     x,
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

// Sum to 1
func Normalization(data []float64) {
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
				count++
			}

			if count >= size {
				break
			}
		}

		if count < size {
			*features = DeleteFromString((*features)[i], *features)
		}

		if len(*features) <= length-size {
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
