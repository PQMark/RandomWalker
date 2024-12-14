package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"sort"

	//"reflect"

	randomforest "github.com/malaschitz/randomForest"
)

type DatasetParams struct {
	NumSamples    int    `json:"num_samples,omitempty"`
	Digits        []int  `json:"digits,omitempty"`
	FilePath      string `json:"file_path,omitempty"`
	MissingMethod string `json:"missing_method,omitempty"`
}

type ModelParams struct {
	NumIteration int `json:"numIteration"`
	NumFolds     int `json:"numFolds"`
	NumFeatures  int `json:"numFeatures,omitempty"`
	BinSize      int `json:"binSize,omitempty"`
	MaxFeatures  int `json:"maxFeatures,omitempty"`
}

type AdvancedParams struct {
	DefaultOptimization bool    `json:"default_optimization"`
	Ntrees              []int   `json:"Ntrees,omitempty"`
	NumLeaves           []int   `json:"num_leaves,omitempty"`
	MaxDepth            []int   `json:"max_depth,omitempty"`
	InitialThreshold    float64 `json:"InitialThreshold,omitempty"`
	DecayFactor         float64 `json:"decayFactor,omitempty"`
}

type OverallParams struct {
	DataSource     string         `json:"data_source"`
	DatasetParams  DatasetParams  `json:"dataset_params"`
	Method         string         `json:"method"`
	ModelParams    ModelParams    `json:"model_params"`
	AdvancedParams AdvancedParams `json:"advanced_params"`
}

// RShiny
// File Reader
// Try new FS method: mRMR
// Replace the GridSearch with Bayesian Optimizor for faster speed

func main() {

	var dataset *Dataset
	var labels []int
	var h HyperParameters

	flag := false

	// Read the JSON file
	data, err := os.ReadFile("temp/overall_params_R.json")
	if err != nil {
		log.Fatalf("Error reading JSON file: %v", err)
	}

	// Parse the JSON into a map for modification
	var jsonData map[string]interface{}
	err = json.Unmarshal(data, &jsonData)
	if err != nil {
		log.Fatalf("Error parsing JSON: %v", err)
	}

	// Modify the JSON (e.g., update scalar fields in "advanced_params")
	if advancedParams, ok := jsonData["advanced_params"].(map[string]interface{}); ok {
		// Update Ntrees
		if ntrees, ok := advancedParams["Ntrees"].(float64); ok {
			advancedParams["Ntrees"] = []float64{ntrees}
		}
		// Update num_leaves
		if numLeaves, ok := advancedParams["num_leaves"].(float64); ok {
			advancedParams["num_leaves"] = []float64{numLeaves}
		}
		// Update max_depth
		if maxDepth, ok := advancedParams["max_depth"].(float64); ok {
			advancedParams["max_depth"] = []float64{maxDepth}
		}
	}

	modifiedData, err := json.Marshal(jsonData)
	if err != nil {
		log.Fatalf("Error marshaling modified JSON: %v", err)
	}

	// Parse the JSON
	var params OverallParams
	err = json.Unmarshal(modifiedData, &params)
	if err != nil {
		log.Fatalf("Error parsing JSON: %v", err)
	}

	if params.DataSource == "mnist" {
		dataset, labels = PrepareMnistData(params.DatasetParams.NumSamples, params.DatasetParams.Digits)

		flag = true

		// Write to json
		if err := Write2Json(dataset, "MNIST.json"); err != nil {
			fmt.Println("Error writing JSON:", err)
		}

	}

	if params.Method == "mRMR" {
		result := RunmRMR(dataset, labels, params.ModelParams.NumIteration, params.ModelParams.BinSize, params.ModelParams.MaxFeatures)

		filename := "mRMR_FeatureImportance.json"

		getFeatureImportances(dataset, labels, result.FeatureSelected, h, filename)

		if flag {
			//get JSON to python
			cmd := exec.Command("python3", "scripts/visualization.py", filename)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			cmd.Run()
		}
	}

	if params.Method == "Boruta" {

		filename := "Boruta_FeatureImportance.json"

		result, _ := Boruta(dataset, labels, params.ModelParams.NumIteration, params.AdvancedParams.Ntrees[0], params.AdvancedParams.MaxDepth[0],
			params.AdvancedParams.NumLeaves[0])

		// Write to json
		getFeatureImportances(dataset, labels, result, h, filename)

		if flag {
			//get JSON to R
			cmd := exec.Command("python3", "scripts/visualization.py", filename)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			cmd.Run()
		}
	}

	if params.Method == "Recursive Feature Elimination" {

		lr := Lr{
			initialThreshold: params.AdvancedParams.InitialThreshold,
			decayFactor:      params.AdvancedParams.DecayFactor,
		}

		filename := "RFE_FeatureImportance.json"

		trainData, trainLabel, testData, testLabel := SplitTrainTest(dataset, labels, 0.8)

		results := RFE(trainData, testData, trainLabel, testLabel, params.ModelParams.NumIteration, params.AdvancedParams.Ntrees[0],
			params.AdvancedParams.MaxDepth[0], params.AdvancedParams.NumLeaves[0], lr, params.ModelParams.MaxFeatures)

		// Write to json
		f := results[int(float64(len(results))*0.2)]
		getFeatureImportances(dataset, labels, f.Features, h, filename)

		//get JSON to R
		if flag {
			cmd := exec.Command("python3", "scripts/visualization.py", filename)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			cmd.Run()
		}
	}

	if params.Method == "Permutation" {
		filename := "Permutation_FeatureImportance.json"

		trainData, trainLabel, testData, testLabel := SplitTrainTest(dataset, labels, 0.8)

		results := permutation(trainData, testData, trainLabel, testLabel, params.ModelParams.NumIteration, params.AdvancedParams.Ntrees[0],
			params.AdvancedParams.MaxDepth[0], params.AdvancedParams.NumLeaves[0])

		// Write to json
		f := GetTopFeatures(results)
		getFeatureImportances(dataset, labels, f, h, filename)

		//get JSON to python
		if flag {
			cmd := exec.Command("python3", "scripts/visualization.py", filename)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			cmd.Run()
		}

	}

	//TestSyntheziedDataPermute()
	// Boruta:
	// TestSyntheziedData()

	// TestSyntheziedDataRFECV()

	//TestImageRFE()
	// jsonDataPath := "testdata/METABRIC_RNA_Mutation.json"
	// jsonLabelPath := "testdata/METABRIC_RNA_Mutation_labels.json"
	// dataset, labels := ReadJSON(jsonDataPath, jsonLabelPath)
	// fmt.Println(dataset)
	// fmt.Println(labels)
	// TestSyntheziedDataRFE(dataset, labels)

	// TestSyntheziedDataWithOptimization()

	// RFE:
	//TestSyntheziedDataRFE()

	ApplyRFEMNIST(400, []int{1, 2})
	//RealDataRFE()

	// ApplyMRMRMNIST(400, []int{1, 2, 9})

	// RealDatamRMR()
}

// func RealDataRFE() {

// 	err := Write2Json(featureStats, "results.json")
// 	if err != nil {
// 		fmt.Println("Error writing to JSON file:", err)
// 	} else {
// 		fmt.Println("Results successfully written to results.json")
// 	}
// }

func GetTopFeatures(features []FeatureAvgMean) []string {
	// Sort by AvgPermutScore in descending order
	sort.Slice(features, func(i, j int) bool {
		return features[i].AvgPermutScore > features[j].AvgPermutScore
	})

	// Calculate the number of top features to select
	topCount := int(float64(len(features)) * 0.1)
	if topCount < 1 {
		topCount = 1 // Ensure at least one feature is selected
	}

	// Extract the feature names of the top features
	topFeatures := make([]string, topCount)
	for i := 0; i < topCount; i++ {
		topFeatures[i] = features[i].Feature
	}

	return topFeatures
}

func getOptimization(params OverallParams) Optimization {
	var opt Optimization

	// Use default grid
	if params.AdvancedParams.DefaultOptimization {
		opt.Optimize = true
		opt.DefaultGrid = true

		return opt
	} else {

		// Don't perform optimization
		if isInt(params.AdvancedParams.MaxDepth) && isInt(params.AdvancedParams.NumLeaves) && isInt(params.AdvancedParams.Ntrees) {
			opt.Optimize = false
			opt.Default = HyperParameters{
				NTrees:   params.AdvancedParams.Ntrees[0],
				MaxDepth: params.AdvancedParams.MaxDepth[0],
				LeafSize: params.AdvancedParams.NumLeaves[0],
			}

			return opt
		} else {

			// Use specified grid
			opt.Optimize = true
			opt.HyperParamsGrid = getCombination(params.AdvancedParams.Ntrees, params.AdvancedParams.NumLeaves, params.AdvancedParams.MaxDepth)

			return opt
		}

	}
}

func isInt(lst []int) bool {
	return len(lst) == 1
}

func getCombination(T, L, D []int) []HyperParameters {
	var grid []HyperParameters

	for _, t := range T {
		for _, l := range L {
			for _, d := range D {
				hyper := HyperParameters{
					NTrees:   t,
					LeafSize: l,
					MaxDepth: d,
				}

				grid = append(grid, hyper)
			}
		}
	}

	return grid
}

func getFeatureImportances(d *Dataset, l []int, f []string, h HyperParameters, filename string) {

	if h.NTrees == 0 {
		h.NTrees = 500
	}

	featureImportances := make(map[string]float64)

	x := ConvertData(d, f)

	forestWithFeatures := randomforest.Forest{
		Data: randomforest.ForestData{
			X:     x,
			Class: l,
		},
		MaxDepth: h.MaxDepth,
		LeafSize: h.LeafSize,
	}
	forestWithFeatures.Train(h.NTrees)

	// Record feature importances
	for i, featureName := range f {
		featureImportances[featureName] = forestWithFeatures.FeatureImportance[i]
	}

	if err := Write2Json(featureImportances, filename); err != nil {
		fmt.Printf("Error writing JSON: %v\n", err)
	}
}

func ApplyMRMRMNIST(num int, features []int) {
	d, l := PrepareMnistData(num, features)

	if err := Write2Json(d, "MNIST.json"); err != nil {
		fmt.Println("Error writing JSON:", err)
	}

	binSize := 1000

	ten := 10
	fourhundred := 400

	results := RunmRMR(d, l, ten, binSize, fourhundred)

	fmt.Println(results)
	//fmt.Println(results.FeatureSelected)
	//fmt.Println(results.F1)
}

func ApplyRFEMNIST(num int, features []int) {
	d, l := PrepareMnistData(num, features)

	if err := Write2Json(d, "MNIST.json"); err != nil {
		fmt.Println("Error writing JSON:", err)
	}

	numIteration := 50
	numFolds := 5

	results := RunRFE(d, l, numIteration, numFolds, 30, Optimization{Default: HyperParameters{
		NTrees: 150,
	}}, Lr{
		initialThreshold: 0.5,
		decayFactor:      1.2,
	})

	maxStats := FindMaxAvgF1(results)
	for _, stat := range maxStats {
		fmt.Printf("Fold %d: %+v\n", stat.FoldIndex, stat.MaxFeature, stat.NumFeatures)
	}

	for i, fold := range results {
		filename := fmt.Sprintf("results_rfe_fold_%d.json", i)
		err := Write2Json(fold, filename)
		if err != nil {
			fmt.Printf("Error writing to %s: %v\n", filename, err)
		} else {
			fmt.Printf("Results successfully written to %s\n", filename)
		}
	}

	// modes := []int{50, 100, 150, 300}
	// featureImportances := make(map[int]map[string]float64)

	// for _, mode := range modes {
	// 	selectedFeatures := getFeaturesRFE(results, mode, 0)

	// 	featureImportances[mode] = make(map[string]float64)

	// 	x := ConvertData(d, selectedFeatures.Features)

	// 	forestWithFeatures := randomforest.Forest{
	// 		Data: randomforest.ForestData{
	// 			X:     x,
	// 			Class: l,
	// 		},
	// 	}
	// 	forestWithFeatures.Train(300)

	// 	// Record feature importances
	// 	for i, featureName := range selectedFeatures.Features {
	// 		featureImportances[mode][featureName] = forestWithFeatures.FeatureImportance[i]
	// 	}
	// }

	// for mode, importanceMap := range featureImportances {
	// 	filename := fmt.Sprintf("MNIST_FeatureImportances_%d.json", mode)
	// 	if err := Write2Json(importanceMap, filename); err != nil {
	// 		fmt.Printf("Error writing JSON for mode %d: %v\n", mode, err)
	// 	}
	// }

	//get JSON to python
	// cmd := exec.Command("python3", "scripts/visualization.py", "MNIST_FeatureImportances_50.json")

	// cmd.Stdout = os.Stdout
	// cmd.Stderr = os.Stderr
	// cmd.Run()

}

/*
func RealDatamRMR() {
	filePath := "/Users/pengqiu/Desktop/GO/src/RandomWalker/testdata/METABRIC_RNA_Mutation.csv"
	colFeatures := true
	irrelevantCols := "2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
	irrelevantRows := ""
	featureIndex := 1
	groupIndex := 5
	dataset, labels := readCSV(filePath, colFeatures, irrelevantCols, irrelevantRows, featureIndex, groupIndex)

	// dataset, labels := createToyDataset()
	fmt.Println(labels)

	numIteration := 30
	numFolds := 5
	binSize := 15
	maxFeatures := 20

	results := RunmRMR(dataset, labels, numIteration, numFolds, binSize, maxFeatures)

	fmt.Println(results)
}
*/

func RealDataRFE() {
	jsonDataPath := "testdata/METABRIC_RNA_Mutation.json"
	jsonLabelPath := "testdata/METABRIC_RNA_Mutation_labels.json"

	jsonData, err := os.ReadFile(jsonDataPath)
	if err != nil {
		fmt.Printf("error unmarshalling JSON data: %v", err)
	}

	jsonLabel, err := os.ReadFile(jsonLabelPath)
	if err != nil {
		fmt.Printf("error unmarshalling JSON data: %v", err)
	}

	var dataset Dataset
	if err := json.Unmarshal(jsonData, &dataset); err != nil {
		fmt.Printf("error unmarshalling JSON data: %v", err)
	}

	var labels []int
	if err := json.Unmarshal(jsonLabel, &labels); err != nil {
		fmt.Printf("error unmarshalling JSON data: %v", err)
	}

	numIteration := 30
	numFolds := 5

	results := RunRFE(&dataset, labels, numIteration, numFolds, 1, Optimization{Default: HyperParameters{
		NTrees: 150,
	}}, Lr{
		initialThreshold: 0.6,
		decayFactor:      1.5,
	})

	for i, fold := range results {
		filename := fmt.Sprintf("METABRIC_results_rfe_fold_%d.json", i)
		err := Write2Json(fold, filename)
		if err != nil {
			fmt.Printf("Error writing to %s: %v\n", filename, err)
		} else {
			fmt.Printf("Results successfully written to %s\n", filename)
		}
	}

	maxStats := FindMaxAvgF1(results)
	for _, stat := range maxStats {
		fmt.Printf("Fold %d: %+v\n", stat.FoldIndex, stat.MaxFeature, stat.NumFeatures)
	}
	filename := fmt.Sprintf("maxStats_rfe.json")
	error := Write2Json(maxStats, filename)
	if error != nil {
		fmt.Printf("Error writing to %s: %v\n", filename, err)
	} else {
		fmt.Printf("Results successfully written to %s\n", filename)
	}

}

func RealDataPermute() {
	jsonDataPath := "temp/METABRIC_RNA_Mutation.json"
	jsonLabelPath := "temp/METABRIC_RNA_Mutation_labels.json"

	jsonData, err := os.ReadFile(jsonDataPath)
	if err != nil {
		fmt.Printf("error unmarshalling JSON data: %v", err)
	}

	jsonLabel, err := os.ReadFile(jsonLabelPath)
	if err != nil {
		fmt.Printf("error unmarshalling JSON data: %v", err)
	}

	var dataset Dataset
	if err := json.Unmarshal(jsonData, &dataset); err != nil {
		fmt.Printf("error unmarshalling JSON data: %v", err)
	}

	var labels []int
	if err := json.Unmarshal(jsonLabel, &labels); err != nil {
		fmt.Printf("error unmarshalling JSON data: %v", err)
	}

	numIteration := 20
	numEstimators := 500
	maxDepth := 15
	numLeaves := 2
	lrParams := Lr{initialThreshold: 0.5, decayFactor: 1.2}
	numFeatures := 1

	train, label, test, tLabel := SplitTrainTest(&dataset, labels, 0.75)

	results := RFE(train, test, label, tLabel, numIteration, numEstimators, maxDepth, numLeaves, lrParams, numFeatures)

	Write2Json(results, "Permutation_METABRIC_RNA_Mutation_20_500_15_2.json")

	fmt.Println(results)
}

func RealDataBoruta() {
	filePath := "/Users/pengqiu/Desktop/GO/src/RandomWalker/testdata/METABRIC_RNA_Mutation.csv"
	colFeatures := true
	irrelevantCols := "2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
	irrelevantRows := ""
	featureIndex := 1
	groupIndex := 5
	dataset, labels := readCSV(filePath, colFeatures, irrelevantCols, irrelevantRows, featureIndex, groupIndex)

	// dataset, labels := createToyDataset()
	fmt.Println(labels)

	numIteration := 50
	numFolds := 5

	//Default: HyperParameters{LeafSize: 0, MaxDepth: 0, NTrees: 1000}

	results := RunBoruta(dataset, labels, numIteration, numFolds, Optimization{Default: HyperParameters{LeafSize: 0, MaxDepth: 0, NTrees: 1000}})

	fmt.Println(results)
}

func createToyDataset() (*Dataset, []int) {
	numInstances := 1000
	dataset := &Dataset{
		Features: []string{"x1", "x2", "x3", "x4", "x5", "noise1", "noise2"},
		Instance: []*Instance{},
		Label:    "label",
	}

	var labels []int

	for i := 0; i < numInstances; i++ {
		x1 := rand.NormFloat64() // Important feature
		x2 := rand.NormFloat64() // Important feature
		x3 := rand.NormFloat64() // Important feature

		// Redundant features (linear combinations)
		x4 := x1 + x2 // Redundant feature
		x5 := x2 - x3 // Redundant feature

		// Noise features
		noise1 := rand.NormFloat64() + 3
		noise2 := rand.NormFloat64() - 1

		// Target variable (binary classification)
		// Let's assume that if (x1 + x2 + x3) > threshold, label is 1 else 0
		threshold := 0.0
		sum := x1 + x2 + x3
		label := 0
		if sum > threshold {
			label = 1
		}

		// Create an instance
		instance := &Instance{
			Features: map[string]float64{
				"x1":     x1,
				"x2":     x2,
				"x3":     x3,
				"x4":     x4,
				"x5":     x5,
				"noise1": noise1,
				"noise2": noise2,
			},
			Label: fmt.Sprintf("%d", label),
		}

		// Add instance and label to dataset
		dataset.Instance = append(dataset.Instance, instance)
		labels = append(labels, label)
	}

	return dataset, labels
}

func TestSyntheziedDataWithOptimization() {
	dataset, label := createToyDataset()

	// Nested CV

	// 5 fold CV
	numFolds := 5
	results := make([]FeaturesF1, numFolds)

	dataFolds, labelFolds := FoldSplit(dataset, label, numFolds)

	for i := 0; i < numFolds; i++ {
		// Get Inner train & Outer test
		innerTrain, innerLabel, outerTest, outerLabel := GetFoldData(dataFolds, labelFolds, i)

		// 10 fold CV for inner train (HP Optimization)
		// Define grid search space
		hyperGrid := hyperparameterGridBoruta(int(float64(len(innerLabel)) * 0.9))
		bestParams, bestF1 := GridSearchParallel(innerTrain, innerLabel, 10, 8, hyperGrid)

		fmt.Printf("Best Hyperparameters - NTrees: %d, MaxDepth: %d, LeafSize: %d\n", bestParams.NTrees, bestParams.MaxDepth, bestParams.LeafSize)
		fmt.Printf("Best F1 Score: %.2f\n", bestF1)

		// Use the tuned HPs for RF in Boruta
		featureSelected, _ := Boruta(innerTrain, innerLabel, 50, bestParams.NTrees, bestParams.MaxDepth, bestParams.LeafSize)

		// Train a RF with selected features with the tuned HPs
		innerTrainProcessed := ConvertData(innerTrain, featureSelected)

		forest := randomforest.Forest{
			Data: randomforest.ForestData{
				X:     innerTrainProcessed,
				Class: innerLabel,
			},
			MaxDepth: bestParams.MaxDepth,
			LeafSize: bestParams.LeafSize,
		}
		forest.Train(bestParams.NTrees)

		// Evaluate the model on outer test
		outerTestProcessed := ConvertData(outerTest, outerTest.Features)
		predictions := Predict(&forest, outerTestProcessed)
		f1 := GetF1Score(predictions, outerLabel)

		results[i] = FeaturesF1{
			FeatureSelected: featureSelected,
			F1:              f1,
		}
	}

	fmt.Println(results)
}

func TestSyntheziedData() {
	numInstances := 1000
	dataset := &Dataset{
		Features: []string{"x1", "x2", "x3", "x4", "x5", "noise1", "noise2"},
		Instance: []*Instance{},
		Label:    "label",
	}

	var labels []int

	for i := 0; i < numInstances; i++ {
		x1 := rand.NormFloat64() // Important feature
		x2 := rand.NormFloat64() // Important feature
		x3 := rand.NormFloat64() // Important feature

		// Redundant features (linear combinations)
		x4 := x1 + x2 // Redundant feature
		x5 := x2 - x3 // Redundant feature

		// Noise features
		noise1 := rand.NormFloat64() + 3
		noise2 := rand.NormFloat64() - 1

		// Target variable (binary classification)
		// Let's assume that if (x1 + x2 + x3) > threshold, label is 1 else 0
		threshold := 0.0
		sum := x1 + x2 + x3
		label := 0
		if sum > threshold {
			label = 1
		}

		// Create an instance
		instance := &Instance{
			Features: map[string]float64{
				"x1":     x1,
				"x2":     x2,
				"x3":     x3,
				"x4":     x4,
				"x5":     x5,
				"noise1": noise1,
				"noise2": noise2,
			},
			Label: fmt.Sprintf("%d", label),
		}

		// Add instance and label to dataset
		dataset.Instance = append(dataset.Instance, instance)
		labels = append(labels, label)
	}

	// Now, call your Boruta function
	numIteration := 50
	numEstimators := 500
	// alpha := 0.05
	maxDepth := 0
	numLeaves := 0

	selectedFeatures, finalResult := Boruta(dataset, labels, numIteration, numEstimators, maxDepth, numLeaves)

	fmt.Println("Selected Features:", selectedFeatures)
	fmt.Println("Results:", finalResult)
}

// Image

// func TestImage() {
// 	num := 80
// 	features := []int{
// 		1, 2,
// 	}
// 	d, l := PrepareMnistData(num, features)
// 	fmt.Println(l)

// 	if err := Write2Json(d, "MNIST.json"); err != nil {
// 		fmt.Println("Error writing JSON:", err)
// 	}

// 	maxDepth := 0
// 	numLeaves := 0

// 	selectedFeatures, finalResult, featureImportances := Boruta(d, l, 50, 150, maxDepth, numLeaves)

// 	//fmt.Println(selectedFeatures)
// 	fmt.Println(finalResult)
// 	fmt.Println(featureImportances)

// 	//transcrib to JSON
// 	if err := Write2Json(featureImportances, "MNIST_Output.json"); err != nil {
// 		fmt.Println("Error writing JSON:", err)
// 	}

// 	//get JSON to python
// 	cmd := exec.Command("python3", "scripts/visualization.py")
// 	cmd.Stdout = os.Stdout
// 	cmd.Stderr = os.Stderr
// 	cmd.Run()

// }

func TestImageRFE() {
	num := 80
	features := []int{
		1, 2, 3, 4, 5, // Example feature set, adjust based on your data
	}
	d, l := PrepareMnistData(num, features)
	fmt.Println(l)

	if err := Write2Json(d, "MNIST.json"); err != nil {
		fmt.Println("Error writing JSON:", err)
	}

	maxDepth := 0
	numLeaves := 0

	selectedFeatures, finalResult := Boruta(d, l, 50, 150, maxDepth, numLeaves)

	featureImportances := make(map[string]float64)
	x := ConvertData(d, selectedFeatures)
	forestWithFeatures := randomforest.Forest{
		Data: randomforest.ForestData{
			X:     x,
			Class: l,
		},
	}
	forestWithFeatures.Train(300)

	for i := 0; i < len(selectedFeatures); i++ {
		featureName := selectedFeatures[i] //d.Features[i]
		featureImportances[featureName] = forestWithFeatures.FeatureImportance[i]
	}

	fmt.Println(selectedFeatures)
	fmt.Println(finalResult)
	fmt.Println(featureImportances)

	//transcrib to JSON
	if err := Write2Json(featureImportances, "MNIST_Output.json"); err != nil {
		fmt.Println("Error writing JSON:", err)
	}

	// Call the Python script for visualization
	cmd := exec.Command("python3", "scripts/visualization.py")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Run()

}

// func TestSyntheziedDataRFE() {
// 	numInstances := 1000
// 	dataset := &Dataset{
// 		Features: []string{"x1", "x2", "x3", "x4", "x5", "noise1", "noise2"},
// 		Instance: []*Instance{},
// 		Label:    "label",
// 	}

// 	var labels []int

// 	for i := 0; i < numInstances; i++ {
// 		x1 := rand.NormFloat64() // Important feature
// 		x2 := rand.NormFloat64() // Important feature
// 		x3 := rand.NormFloat64() // Important feature

// 		// Redundant features (linear combinations)
// 		x4 := x1 + x2 // Redundant feature
// 		x5 := x2 - x3 // Redundant feature

// 		// Noise features
// 		noise1 := rand.NormFloat64() + 3
// 		noise2 := rand.NormFloat64() - 1

// 		// Target variable (binary classification)
// 		// Let's assume that if (x1 + x2 + x3) > threshold, label is 1 else 0
// 		threshold := 0.0
// 		sum := x1 + x2 + x3
// 		label := 0
// 		if sum > threshold {
// 			label = 1
// 		}

// 		// Create an instance
// 		instance := &Instance{
// 			Features: map[string]float64{
// 				"x1":     x1,
// 				"x2":     x2,
// 				"x3":     x3,
// 				"x4":     x4,
// 				"x5":     x5,
// 				"noise1": noise1,
// 				"noise2": noise2,
// 			},
// 			Label: fmt.Sprintf("%d", label),
// 		}

// 		// Add instance and label to dataset
// 		dataset.Instance = append(dataset.Instance, instance)
// 		labels = append(labels, label)
// 	}

// 	// Now, call your Boruta function
// 	numIteration := 50
// 	numEstimators := 100
// 	// alpha := 0.05
// 	maxDepth := 0
// 	numLeaves := 0

// 	train, label, test, tLabel := SplitTrainTest(dataset, labels, 0.75)
// 	featureStats := REF(train, test, label, tLabel, numIteration, numEstimators, maxDepth, numLeaves)

// 	fmt.Println("Results:", featureStats)

// 	err := Write2Json(featureStats, "results.json")
// 	if err != nil {
// 		fmt.Println("Error writing to JSON file:", err)
// 	} else {
// 		fmt.Println("Results successfully written to results.json")
// 	}
// }

func TestSyntheziedDataRFE() {
	numInstances := 1000
	dataset := &Dataset{
		Features: []string{"x1", "x2", "x3", "x4", "x5", "noise1", "noise2"},
		Instance: []*Instance{},
		Label:    "label",
	}

	var labels []int

	for i := 0; i < numInstances; i++ {
		x1 := rand.NormFloat64() // Important feature
		x2 := rand.NormFloat64() // Important feature
		x3 := rand.NormFloat64() // Important feature

		// Redundant features (linear combinations)
		x4 := x1 + x2 // Redundant feature
		x5 := x2 - x3 // Redundant feature

		// Noise features
		noise1 := rand.NormFloat64() + 3
		noise2 := rand.NormFloat64() - 1

		// Target variable (binary classification)
		// Let's assume that if (x1 + x2 + x3) > threshold, label is 1 else 0
		threshold := 0.0
		sum := x1 + x2 + x3
		label := 0
		if sum > threshold {
			label = 1
		}

		// Create an instance
		instance := &Instance{
			Features: map[string]float64{
				"x1":     x1,
				"x2":     x2,
				"x3":     x3,
				"x4":     x4,
				"x5":     x5,
				"noise1": noise1,
				"noise2": noise2,
			},
			Label: fmt.Sprintf("%d", label),
		}

		// Add instance and label to dataset
		dataset.Instance = append(dataset.Instance, instance)
		labels = append(labels, label)
	}

	// Now, call your Boruta function
	numIteration := 50
	numEstimators := 100
	// alpha := 0.05
	maxDepth := 0
	numLeaves := 0

	train, trainLabels, test, testLabels := SplitTrainTest(dataset, labels, 0.75)
	featureStats := RFE(train, test, trainLabels, testLabels, numIteration, numEstimators, maxDepth, numLeaves, Lr{}, 1)

	fmt.Println("Results:", featureStats)
	// fmt.Println("Importances:", featureImportances)

	err := Write2Json(featureStats, "results.json")
	if err != nil {
		fmt.Println("Error writing to JSON file:", err)
	} else {
		fmt.Println("Results successfully written to results.json")
	}
}

// func TestSyntheziedDataRFECV() {
// 	numInstances := 1000
// 	dataset := &Dataset{
// 		Features: []string{"x1", "x2", "x3", "x4", "x5", "noise1", "noise2"},
// 		Instance: []*Instance{},
// 		Label:    "label",
// 	}

// 	var labels []int

// 	for i := 0; i < numInstances; i++ {
// 		x1 := rand.NormFloat64() // Important feature
// 		x2 := rand.NormFloat64() // Important feature
// 		x3 := rand.NormFloat64() // Important feature

// 		// Redundant features (linear combinations)
// 		x4 := x1 + x2 // Redundant feature
// 		x5 := x2 - x3 // Redundant feature

// 		// Noise features
// 		noise1 := rand.NormFloat64() + 3
// 		noise2 := rand.NormFloat64() - 1

// 		// Target variable (binary classification)
// 		// Let's assume that if (x1 + x2 + x3) > threshold, label is 1 else 0
// 		threshold := 0.0
// 		sum := x1 + x2 + x3
// 		label := 0
// 		if sum > threshold {
// 			label = 1
// 		}

// 		// Create an instance
// 		instance := &Instance{
// 			Features: map[string]float64{
// 				"x1":     x1,
// 				"x2":     x2,
// 				"x3":     x3,
// 				"x4":     x4,
// 				"x5":     x5,
// 				"noise1": noise1,
// 				"noise2": noise2,
// 			},
// 			Label: fmt.Sprintf("%d", label),
// 		}

// 		// Add instance and label to dataset
// 		dataset.Instance = append(dataset.Instance, instance)
// 		labels = append(labels, label)
// 	}

// 	// Now, call your Boruta function
// 	numIteration := 50
// 	numEstimators := 100
// 	// alpha := 0.05
// 	maxDepth := 0
// 	numLeaves := 0

// 	//train, label, test, tLabel := SplitTrainTest(dataset, labels, 0.75)
// 	optimalFeatureCount := RFECV(dataset, labels, numIteration, numEstimators, maxDepth, numLeaves, 5, 0)

// 	fmt.Println("Optimal Feature Count: ", optimalFeatureCount)

// err := Write2Json(featureStats, "results.json")
// if err != nil {
// 	fmt.Println("Error writing to JSON file:", err)
// } else {
// 	fmt.Println("Results successfully written to results.json")
// }

// func TestSyntheziedDataRFEFromCSV(filePath string) {
// 	// Open the CSV file
// 	file, err := os.Open(filePath)
// 	if err != nil {
// 		fmt.Println("Error opening CSV file:", err)
// 		return
// 	}
// 	defer file.Close()

// 	// Read the CSV file
// 	reader := csv.NewReader(file)
// 	rows, err := reader.ReadAll()
// 	if err != nil {
// 		fmt.Println("Error reading CSV file:", err)
// 		return
// 	}

// 	features := rows[0][1:]

// 	// Initialize Dataset and labels
// 	dataset := &Dataset{
// 		Features: features,
// 		Instance: []*Instance{},
// 		Label:    "label",
// 	}
// 	var labels []int

// 	// Parse rows into dataset
// 	for _, row := range rows[1:] { // Skip header row
// 		// Parse the label (column 1)
// 		// label, err := strconv.Atoi(row[0])
// 		// if err != nil {
// 		// 	fmt.Println("Error parsing label:", err)
// 		// 	continue
// 		// }

// 		// Parse features (columns 2 onward)
// 		featureMap := make(map[string]float64)
// 		for i, feature := range features {
// 			value, err := strconv.ParseFloat(row[i], 64)
// 			if err != nil {
// 				fmt.Println("Error parsing feature value:", err)
// 				continue
// 			}
// 			featureMap[feature] = value
// 		}

// 		// binary classification of relative liver weights
// 		var mean float64
// 		mean = calculateMeanLiverWeight(filePath)
// 		fmt.Println(mean)
// 		var label int
// 		val, _ := strconv.ParseFloat(row[0], 64)
// 		fmt.Println(val)
// 		if val >= mean {
// 			label = 1
// 		} else {
// 			label = 0
// 		}

// 		// Create an instance
// 		instance := &Instance{
// 			Features: featureMap,
// 			Label:    fmt.Sprintf("%d", label),
// 		}

// 		// Add to dataset and labels
// 		dataset.Instance = append(dataset.Instance, instance)
// 		labels = append(labels, label)
// 	}

// 	TestSyntheziedDataRFE(dataset, labels)

// }

// func TestSyntheziedDataRFE(dataset *Dataset, labels []int) {
// 	// 	numInstances := 1000
// 	// 	dataset := &Dataset{
// 	// 		Features: []string{
// 	// 			"x1", "x2", "x3", "x4", "x5", "noise1", "noise2", "redundant1", "redundant2",
// 	// 			"redundant3", "redundant4", "redundant5", "redundant6", "noise3", "noise4",
// 	// 			"noise5", "noise6", "noise7", "noise8", "noise9"},
// 	// 		Instance: []*Instance{},
// 	// 		Label:    "label",
// 	// 	}

// 	// 	var labels []int

// 	// 	for i := 0; i < numInstances; i++ {
// 	// 		// Important features
// 	// 		x1 := rand.NormFloat64()
// 	// 		x2 := rand.NormFloat64()
// 	// 		x3 := rand.NormFloat64()

// 	// 		// Redundant features (linear combinations)
// 	// 		x4 := x1 + x2
// 	// 		x5 := x2 - x3
// 	// 		redundant1 := x1 * x2
// 	// 		redundant2 := x3 * x2
// 	// 		redundant3 := x1 * x3
// 	// 		redundant4 := x1 + x3
// 	// 		redundant5 := x1 - x2

// 	// 		redundant6 := x2 + x3

// 	// 		// Noise features
// 	// 		noise1 := rand.NormFloat64() + 3
// 	// 		noise2 := rand.NormFloat64() - 1
// 	// 		noise3 := rand.NormFloat64()
// 	// 		noise4 := rand.NormFloat64()
// 	// 		noise5 := rand.NormFloat64()
// 	// 		noise6 := rand.NormFloat64()
// 	// 		noise7 := rand.NormFloat64()
// 	// 		noise8 := rand.NormFloat64()
// 	// 		noise9 := rand.NormFloat64()

// 	// 		// Target variable (binary classification)
// 	// 		threshold := 0.0
// 	// 		sum := x1 + x2 + x3
// 	// 		label := 0
// 	// 		if sum > threshold {
// 	// 			label = 1
// 	// 		}

// 	// 		// Create an instance
// 	// 		instance := &Instance{
// 	// 			Features: map[string]float64{
// 	// 				"x1":         x1,
// 	// 				"x2":         x2,
// 	// 				"x3":         x3,
// 	// 				"x4":         x4,
// 	// 				"x5":         x5,
// 	// 				"noise1":     noise1,
// 	// 				"noise2":     noise2,
// 	// 				"redundant1": redundant1,
// 	// 				"redundant2": redundant2,
// 	// 				"redundant3": redundant3,
// 	// 				"redundant4": redundant4,
// 	// 				"redundant5": redundant5,
// 	// 				"redundant6": redundant6,
// 	// 				"noise3":     noise3,
// 	// 				"noise4":     noise4,
// 	// 				"noise5":     noise5,
// 	// 				"noise6":     noise6,
// 	// 				"noise7":     noise7,
// 	// 				"noise8":     noise8,
// 	// 				"noise9":     noise9,
// 	// 			},
// 	// 			Label: fmt.Sprintf("%d", label),
// 	// 		}

// 	// 		// Add instance and label to dataset
// 	// 		dataset.Instance = append(dataset.Instance, instance)
// 	// 		labels = append(labels, label)
// 	// 	}

// 	// Now, call your RFE function
// 	numIteration := 20
// 	numEstimators := 500
// 	maxDepth := 10
// 	numLeaves := 2
// 	//minFeatures := 2  // Minimum feature count to stop at

// 	train, trainLabels, test, testLabels := SplitTrainTest(dataset, labels, 0.75)
// 	featureStats := RFE(train, test, trainLabels, testLabels, numIteration, numEstimators, maxDepth, numLeaves)

// 	// Print Results
// 	fmt.Println("breast_cancer_Results:", featureStats)

// 	// Write results to JSON file
// 	err := Write2Json(featureStats, "results.json")
// 	if err != nil {
// 		fmt.Println("Error writing to JSON file:", err)
// 	} else {
// 		fmt.Println("Results successfully written to results.json")
// 	}
// }

func TestSyntheziedDataPermute() {
	numInstances := 1000
	dataset := &Dataset{
		Features: []string{"x1", "x2", "x3", "x4", "x5", "noise1", "noise2"},
		Instance: []*Instance{},
		Label:    "label",
	}

	var labels []int

	for i := 0; i < numInstances; i++ {
		x1 := rand.NormFloat64() // Important feature
		x2 := rand.NormFloat64() // Important feature
		x3 := rand.NormFloat64() // Important feature

		// Redundant features (linear combinations)
		x4 := x1 + x2 // Redundant feature
		x5 := x2 - x3 // Redundant feature

		// Noise features
		noise1 := rand.NormFloat64() + 3
		noise2 := rand.NormFloat64() - 1

		// Target variable (binary classification)
		// Let's assume that if (x1 + x2 + x3) > threshold, label is 1 else 0
		threshold := 0.0
		sum := x1 + x2 + x3
		label := 0
		if sum > threshold {
			label = 1
		}

		// Create an instance
		instance := &Instance{
			Features: map[string]float64{
				"x1":     x1,
				"x2":     x2,
				"x3":     x3,
				"x4":     x4,
				"x5":     x5,
				"noise1": noise1,
				"noise2": noise2,
			},
			Label: fmt.Sprintf("%d", label),
		}

		// Add instance and label to dataset
		dataset.Instance = append(dataset.Instance, instance)
		labels = append(labels, label)
	}

	// Now, call your Boruta function
	numIteration := 100
	numEstimators := 300
	// alpha := 0.05
	maxDepth := 0
	numLeaves := 0

	train, label, test, tLabel := SplitTrainTest(dataset, labels, 0.75)
	results := permutation(train, test, label, tLabel, numIteration, numEstimators, maxDepth, numLeaves)

	fmt.Println("Results:", results)
}
