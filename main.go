package main

import (
	"fmt"
	"math/rand"
	"os"
	"encoding/json"
	"os/exec"

	randomforest "github.com/malaschitz/randomForest"
)

type FeaturesF1 struct {
	FeatureSelected []string
	F1              float64
}

// RShiny
// File Reader
// Try new FS method: mRMR
// Replace the GridSearch with Bayesian Optimizor for faster speed


func main() {

	//TestSyntheziedDataPermute()
	// Boruta:
	// TestSyntheziedData()

	// TestImage()

	// TestSyntheziedDataWithOptimization()

	// RFE:
	//TestSyntheziedDataRFE()


	// ApplyRFEMNIST(400, []int{1, 2})

	cmd := exec.Command("python3", "scripts/visualization.py", "MNIST_FeatureImportances_300.json")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Run()

	// RealDatamRMR()
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
		InitialThreshold: 0.5,
		decayFactor: 1.2,
	})

	modes := []int{50, 100, 150, 300}
	featureImportances := make(map[int]map[string]float64)

	for _, mode := range modes {
		selectedFeatures := getFeaturesRFE(results, mode, 0)

		featureImportances[mode] = make(map[string]float64)

		x := ConvertData(d, selectedFeatures.Features)

		forestWithFeatures := randomforest.Forest{
			Data: randomforest.ForestData{
				X:    x,
				Class: l,
			},
		}
		forestWithFeatures.Train(300)

		// Record feature importances
		for i, featureName := range selectedFeatures.Features {
			featureImportances[mode][featureName] = forestWithFeatures.FeatureImportance[i]
		}
	}

	for mode, importanceMap := range featureImportances {
		filename := fmt.Sprintf("MNIST_FeatureImportances_%d.json", mode)
		if err := Write2Json(importanceMap, filename); err != nil {
			fmt.Printf("Error writing JSON for mode %d: %v\n", mode, err)
		}
	}

	//get JSON to python
	cmd := exec.Command("python3", "scripts/visualization.py")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Run()

}

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
	lrParams := Lr{InitialThreshold: 0.5, decayFactor: 1.2}
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
	numEstimators := 100
	// alpha := 0.05
	maxDepth := 0
	numLeaves := 0

	selectedFeatures, finalResult := Boruta(dataset, labels, numIteration, numEstimators, maxDepth, numLeaves)

	fmt.Println("Selected Features:", selectedFeatures)
	fmt.Println("Results:", finalResult)
}


// Image
func TestImage() {
	num := 80
	features := []int{
		1, 2,
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
			X: x,
			Class: l,
		},
	}
	forestWithFeatures.Train(300)

	for i := 0; i < len(selectedFeatures); i ++ {
		featureName := selectedFeatures[i]		//d.Features[i]
		featureImportances[featureName] = forestWithFeatures.FeatureImportance[i]
	}

	fmt.Println(selectedFeatures)
	fmt.Println(finalResult)
	fmt.Println(featureImportances)

	//transcrib to JSON
	if err := Write2Json(featureImportances, "MNIST_Output.json"); err != nil {
		fmt.Println("Error writing JSON:", err)
	}

	//get JSON to python
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
		Features: []string{
			"x1", "x2", "x3", "x4", "x5", "noise1", "noise2", "redundant1", "redundant2",
			"redundant3", "redundant4", "redundant5", "redundant6", "noise3", "noise4",
			"noise5", "noise6", "noise7", "noise8", "noise9"},
		Instance: []*Instance{},
		Label:    "label",
	}

	var labels []int

	for i := 0; i < numInstances; i++ {
		// Important features
		x1 := rand.NormFloat64()
		x2 := rand.NormFloat64()
		x3 := rand.NormFloat64()

		// Redundant features (linear combinations)
		x4 := x1 + x2
		x5 := x2 - x3
		redundant1 := x1 * x2
		redundant2 := x3 * x2
		redundant3 := x1 * x3
		redundant4 := x1 + x3
		redundant5 := x1 - x2

		redundant6 := x2 + x3

		// Noise features
		noise1 := rand.NormFloat64() + 3
		noise2 := rand.NormFloat64() - 1
		noise3 := rand.NormFloat64()
		noise4 := rand.NormFloat64()
		noise5 := rand.NormFloat64()
		noise6 := rand.NormFloat64()
		noise7 := rand.NormFloat64()
		noise8 := rand.NormFloat64()
		noise9 := rand.NormFloat64()

		// Target variable (binary classification)
		threshold := 0.0
		sum := x1 + x2 + x3
		label := 0
		if sum > threshold {
			label = 1
		}

		// Create an instance
		instance := &Instance{
			Features: map[string]float64{
				"x1":         x1,
				"x2":         x2,
				"x3":         x3,
				"x4":         x4,
				"x5":         x5,
				"noise1":     noise1,
				"noise2":     noise2,
				"redundant1": redundant1,
				"redundant2": redundant2,
				"redundant3": redundant3,
				"redundant4": redundant4,
				"redundant5": redundant5,
				"redundant6": redundant6,
				"noise3":     noise3,
				"noise4":     noise4,
				"noise5":     noise5,
				"noise6":     noise6,
				"noise7":     noise7,
				"noise8":     noise8,
				"noise9":     noise9,
			},
			Label: fmt.Sprintf("%d", label),
		}

		// Add instance and label to dataset
		dataset.Instance = append(dataset.Instance, instance)
		labels = append(labels, label)
	}

	// Now, call your RFE function
	numIteration := 50
	numEstimators := 100
	maxDepth := 0
	numLeaves := 0
	//minFeatures := 2  // Minimum feature count to stop at

	train, trainLabels, test, testLabels := SplitTrainTest(dataset, labels, 0.75)
	featureStats := RFE(train, test, trainLabels, testLabels, numIteration, numEstimators, maxDepth, numLeaves, Lr{}, 1)

	// Print Results
	fmt.Println("Results:", featureStats)

	// Write results to JSON file
	err := Write2Json(featureStats, "results.json")
	if err != nil {
		fmt.Println("Error writing to JSON file:", err)
	} else {
		fmt.Println("Results successfully written to results.json")
	}
}

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
