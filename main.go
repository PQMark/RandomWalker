package main

import (
	//"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"os/exec"

	//"strconv"

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

	TestSyntheziedDataRFECV()

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

	//filePath := "/Users/pengqiu/Desktop/GO/src/RandomWalker/testdata/Metabolite_name_parkinson.csv"
	// filePath := "/Users/junequ/go/RandomWalker/Metabolite_name_parkinson.csv"
	// colFeatures := false
	// irrelevantCols := "2"
	// irrelevantRows := ""
	// featureIndex := 1
	// groupIndex := 1
	// dataset, labels := readCSV(filePath, colFeatures, irrelevantCols, irrelevantRows, featureIndex, groupIndex)
	//fmt.Println(dataset)
	//fmt.Println(labels)
	//TestSyntheziedDataRFE(dataset, labels)
	// dataset, labels := createToyDataset()

	// numIteration := 50
	// numFolds := 2

	//results := RunBoruta(dataset, labels, numIteration, numFolds, Optimization{Default: HyperParameters{LeafSize: 2, MaxDepth: 15, NTrees: 2000}})

	//fmt.Println(results)

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

	maxDepth := 5   // Example value, adjust as needed
	numLeaves := 10 // Example value, adjust as needed
	numIteration := 50
	numEstimators := 100

	// Use RFE
	train, dlabel, test, tLabel := SplitTrainTest(d, l, 0.75)
	results, featureImportance := RFE(train, test, dlabel, tLabel, numIteration, numEstimators, maxDepth, numLeaves, 300)

	// Print the results for each iteration
	for _, result := range results {
		fmt.Println("Selected Features:", result.Features)
		fmt.Println("Average F1 Score:", result.AvgF1)
		fmt.Println("Standard Error of F1 Score:", result.ErrorF1)
	}

	// Save the feature importances to a JSON file
	if err := Write2Json(featureImportance, "MNIST_Output.json"); err != nil {
		fmt.Println("Error writing JSON:", err)
	}

	// Call the Python script for visualization
	cmd := exec.Command("python3", "scripts/visualization.py")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Run()
}

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

	train, label, test, tLabel := SplitTrainTest(dataset, labels, 0.75)
	featureStats, featureImportances := RFE(train, test, label, tLabel, numIteration, numEstimators, maxDepth, numLeaves, 4)

	fmt.Println("Results:", featureStats)
	fmt.Println("Importances:", featureImportances)

	err := Write2Json(featureStats, "results.json")
	if err != nil {
		fmt.Println("Error writing to JSON file:", err)
	} else {
		fmt.Println("Results successfully written to results.json")
	}
}

func TestSyntheziedDataRFECV() {
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

	//train, label, test, tLabel := SplitTrainTest(dataset, labels, 0.75)
	optimalFeatureCount := RFECV(dataset, labels, numIteration, numEstimators, maxDepth, numLeaves, 5, 0)

	fmt.Println("Optimal Feature Count: ", optimalFeatureCount)

	// err := Write2Json(featureStats, "results.json")
	// if err != nil {
	// 	fmt.Println("Error writing to JSON file:", err)
	// } else {
	// 	fmt.Println("Results successfully written to results.json")
	// }
}

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
