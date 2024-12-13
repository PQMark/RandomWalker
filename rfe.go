package main

import (
	"fmt"
	"math"

	randomforest "github.com/malaschitz/randomForest"
)

// Input:
// numFolds: at least 2 folds
// lrParams: initial threshold and decay factor (default: 0.2 and 1.5)
// optimization.optimize: true-> optimize for best NTrees, MaxDepth, LeafSize | false-> default hhyper parameters
// Output: list of feature stats for each fold
func RunRFE(data *Dataset, labels []int, numIteration, numFolds, numFeatures int, optimization Optimization, lrParams Lr) [][]FeatureStats {

	fmt.Printf(
		"Running RFECV across %d folds with elimination threshold %.2f and decay factor %.2f\n",
		numFolds, lrParams.initialThreshold, lrParams.decayFactor,
	)
	
	dataFolds, labelFolds := FoldSplit(data, labels, numFolds)
	results := make([][]FeatureStats, numFolds)
	var hyperGrid []HyperParameters
	var hyperParams HyperParameters

	for i := 0; i < numFolds; i++ {
		fmt.Println("starting RFECV for fold", i)

		// Get Inner train & Outer test
		innerTrain, innerLabel, outerTest, outerLabel := GetFoldData(dataFolds, labelFolds, i)

		if optimization.Optimize {
			// Perform Optimization

			// 10 fold CV for inner train (HP Optimization)
			// Define grid search space
			if optimization.DefaultGrid {
				// Use default grid
				hyperGrid = hyperparameterGridBoruta(int(float64(len(innerLabel)) * 0.9))
			} else {
				// User specified
				hyperGrid = optimization.HyperParamsGrid
			}

			bestParams, bestF1 := GridSearchParallel(innerTrain, innerLabel, 10, optimization.numProcs, hyperGrid)
			fmt.Printf("Best Hyperparameters - NTrees: %d, MaxDepth: %d, LeafSize: %d\n", bestParams.NTrees, bestParams.MaxDepth, bestParams.LeafSize)
			fmt.Printf("Best F1 Score: %.2f\n", bestF1)

			hyperParams = bestParams
		} else {
			// Use default value for RFE
			hyperParams = optimization.Default

			if hyperParams.NTrees == 0 {
				hyperParams.NTrees = 1000
			}
		}

		// RFE
		featureStats := RFE(innerTrain, outerTest, innerLabel, outerLabel, numIteration, hyperParams.NTrees, hyperParams.MaxDepth, hyperParams.LeafSize, lrParams, numFeatures)
		fmt.Printf("fold %d completed", i)
		results[i] = featureStats
	}

	return results
}

// Input:
// numFeatures: min number of features to return
// numIterations: train random forest over numIterations for each round of feature elimination
// Output: list of feature stats (each feature count with associated avgF1 and error)
func RFE(d, test *Dataset, dLabel, tLabel []int, numIteration, numEstimators, maxDepth, numLeaves int, lrParams Lr, numFeatures int) []FeatureStats {
	fmt.Println("running RFE...")
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
		run++

		tempResults := make([]float64, 0, numIteration)       // store the F1 score
		featureImportances := make([][]float64, numIteration) // store the importance score

		// Train the RF model numIteration times
		for i := 0; i < numIteration; i++ {
			// fmt.Println("RFE run:", run, "/", i)
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

		// using power law decay
		threshold := FeatureDecayScheduler(&featuresToConsiderCopy, len(d.Features), lrParams)

		// Discard the features
		DiscardFeatures(featureImportances, &featuresToConsider, threshold)

		// Check the number of features remaining
		if len(featuresToConsider) <= numFeatures {
			return results
		}

		fmt.Println(featuresToConsider)
		fmt.Println("number of features to consider:", len(featuresToConsider))
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

// Normalize data to sum to 1
func Normalization(data []float64) {
	sum := 0.0

	for _, val := range data {
		sum += val
	}

	for i := range data {
		data[i] /= sum
	}
}

// Input:
// data: feature importances across each feature count for all numIterations
// threshold: % of features with the lowest FI eliminated, at least 1 feature removed
func DiscardFeatures(data [][]float64, features *[]string, threshold float64) {
	length := len(*features)

	if len(data) == 0 || len(data[0]) == 0 || len(*features) == 0 {
		return
	}

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

		if count < size && len(*features) > 0 {
			*features = DeleteFromString((*features)[i], *features)
		}

		if len(*features) <= length-size {
			break
		}

	}

}

// Input:
// lrParams: default initial threshold : 0.2 and decay factor : 1.5 unless specified
// Output:
// threshold determined by : initialThreshold * (%remaining features)^decayFactor
func FeatureDecayScheduler(features *[]string, numTotalFeatures int, lrParams Lr) float64 {
	// default initial threshold and decay factor
	if lrParams.initialThreshold == 0.0 {
		lrParams.initialThreshold = 0.2
	}

	if lrParams.decayFactor == 0.0 {
		lrParams.decayFactor = 1.5
	}

	var threshold float64
	remainFeatures := len(*features)

	//fmt.Println(numTotalFeatures)
	remainingPercent := float64(remainFeatures) / float64(numTotalFeatures)
	if remainingPercent <= 0.20 {
		threshold = 0.03
		fmt.Println("t: ", threshold)

	} else {
		// fmt.Println(initialThreshold)
		// fmt.Println(decayFactor)
		threshold = float64(lrParams.initialThreshold) * math.Pow(remainingPercent, lrParams.decayFactor)
		fmt.Println("t: ", threshold)
	}
	return threshold
}

// Mode: 0 --> get the best, output those with frequency above binomial threshold(default)
// Mode: otherwise --> pick feature subset with the same size

// threshold: 0 --> use default
// A threshold of 1 will output the feature subset union
func getFeaturesRFE(results [][]FeatureStats, mode, threshold int) FeatureStats {

	if threshold == 0 {
		threshold = CalculateThreshold(len(results))
	}

	counts := make(map[string]int)
	tempResults := make([]float64, 0)

	for i := 0; i < len(results); i++ {
		var selectedSubset []string
		maxScore := 0.0
		minDiff := math.Inf(1)

		for j := 0; j < len(results[i]); j++ {

			if mode == 0 {
				// Get the subset with the highest score
				if results[i][j].AvgF1 > maxScore {
					maxScore = results[i][j].AvgF1
					selectedSubset = results[i][j].Features
				}
			} else {
				// Pick subsets with the size close to 'mode'
				diff := math.Abs(float64(len(results[i][j].Features) - mode))

				if diff < minDiff {
					selectedSubset = results[i][j].Features
					maxScore = results[i][j].AvgF1
					minDiff = diff
				} else if diff > minDiff {
					break
				}
			}
		}

		// Record frequencies of features in the selected subset
		for _, feature := range selectedSubset {
			counts[feature]++
		}

		tempResults = append(tempResults, maxScore)
	}

	avgF1 := Average(tempResults)
	errorF1 := standardError(tempResults, avgF1)

	selectedFeatures := make([]string, 0)
	for feature, frequency := range counts {
		if frequency >= threshold {
			selectedFeatures = append(selectedFeatures, feature)
		}
	}

	return FeatureStats{
		Features: selectedFeatures,
		AvgF1:    avgF1,
		ErrorF1:  errorF1,
	}
}

// Input: results slice containing feature stats for each fold
// returns a slice containing the fold #, number of selected features, best feature stat for each fold
func FindMaxAvgF1(results [][]FeatureStats) []FoldMax {
	var maxStats []FoldMax

	for i, fold := range results {
		if len(fold) == 0 {
			continue
		}

		maxFeature := fold[0]
		for _, feature := range fold {
			if feature.AvgF1 > maxFeature.AvgF1 {
				maxFeature = feature
			}
		}

		numFeatures := len(maxFeature.Features)
		maxStats = append(maxStats, FoldMax{i, maxFeature, numFeatures})
	}

	return maxStats
}