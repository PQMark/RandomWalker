package main

import (
	"fmt"

	"github.com/malaschitz/randomForest"
)

type HyperParameters struct {
	NTrees int
	MaxDepth int 
	LeafSize int
}

type SearchResult struct {
	Params HyperParameters
	F1Avg float64
}

// number of trees
// Leaf size : default 1 (less than 20 entries), 50 (more than 1000 entries), # entries / 20 (Between)
// Max depth : default 10 
// use F1 score for potentially imbalanced dataset 
// Fine 
func GridSearchParallel(data *Dataset, labels []int, numFolds, numProcs int, hyperGrid []HyperParameters) (HyperParameters, float64) {
	dataFolds, labelFolds := FoldSplit(data, labels, numFolds)

	// Define search grid 
	// hyperGrid := hyperparameterGridBoruta(len(labels))

	resultsChan := make(chan SearchResult, len(hyperGrid))
	availability := make(chan bool, numProcs)

	for _, params := range hyperGrid {

		availability <- true

		go GridSearch(params, dataFolds, labelFolds, resultsChan, availability)
	}

	var bestParams HyperParameters 
	bestF1 := -1.0

	for i := 0; i < len(hyperGrid); i++ {
		res := <- resultsChan

		fmt.Println("P", res.Params)
		fmt.Println("F", res.F1Avg)

		fmt.Println(res)

		if res.F1Avg > bestF1 {
			bestF1 = res.F1Avg
			bestParams = res.Params
		}
	}

	return bestParams, bestF1
}	

func hyperparameterGridBoruta(dataLength int) []HyperParameters {
	// nTrees := make([]int, 5)
	nTrees := []int{100, 200, 300, 500, 1000}
	maxDepth := []int{5, 10, 20, 25, 30, 50, 100}
	leafSize := make([]int, 5)

	var defaultLeafSize int
	if dataLength <= 20 {
		defaultLeafSize = 1
	} else if dataLength >= 1000 {
		defaultLeafSize = 50
	} else {
		defaultLeafSize = dataLength / 20
	}

	for i := 0; i < 5; i++ {
		if defaultLeafSize > 1 {
			leafSize[i] = int(float64(defaultLeafSize) * (0.5 + 0.25*float64(i)))
		} else {
			leafSize[i] = defaultLeafSize + i
		}
	}

	/*
	for i := 0; i < 5; i++ {
		nTrees[i] = 100 + i * 500
	}
	*/

	// Get all the combinations
	var grid []HyperParameters
	for _, numTree := range nTrees {
		for _, depth := range maxDepth {
			for _, leaf := range leafSize {
				grid = append(grid, HyperParameters{
					NTrees: numTree,
					MaxDepth: depth,
					LeafSize: leaf,
				})
			}
		}
	}

	return grid
}

func GridSearch(params HyperParameters, dataFolds []*Dataset, labelFolds [][]int, resultChan chan SearchResult, a chan bool) {

	numFolds := len(dataFolds)
	f1Scores := make([]float64, 0)

	for i := 0; i < numFolds; i++ {

		fmt.Println(i, ":Training RF with Params:", params)

		trainData, trainLabel, valData, valLabel := GetFoldData(dataFolds, labelFolds, i)

		// For large dataset, set the number of estimators to 1000
		// selectedFeatures, _, _:= Boruta(trainData, trainLabel, 50, 500)

		// Retain data with all features 
		trainDataProcessed := ConvertToData(trainData, trainData.Features)
		valDataProcessed := ConvertToData(valData, valData.Features)

		forest := randomforest.Forest{
			Data: randomforest.ForestData{
				X: trainDataProcessed,
				Class: trainLabel,
			},
			LeafSize: params.LeafSize,
			MaxDepth: params.MaxDepth,
		}
		forest.Train(params.NTrees)

		// predict on the validation set 
		predictions := Predict(&forest, valDataProcessed)

		// f1 score
		f1 := GetF1Score(predictions, valLabel)

		f1Scores = append(f1Scores, f1)
	}

	// Calculate the average 
	avgF1 := Average(f1Scores)

	// Send results to channel 
	resultChan <- SearchResult{
		Params: params,
		F1Avg: avgF1,
	}

	// release one spot
	<- a 
}

