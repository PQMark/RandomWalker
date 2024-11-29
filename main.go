package main

import (
	"fmt"
	"math/rand"
	"os"
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

	TestSyntheziedDataPermute()
	// Boruta:
	// TestSyntheziedData()

	// TestImage()

	// TestSyntheziedDataWithOptimization()

	// RFE:
	//TestSyntheziedDataRFE()

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
		featureSelected, _, _ := Boruta(innerTrain, innerLabel, 50, bestParams.NTrees, bestParams.MaxDepth, bestParams.LeafSize)

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

	selectedFeatures, finalResult, _ := Boruta(dataset, labels, numIteration, numEstimators, maxDepth, numLeaves)

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

	selectedFeatures, finalResult, featureImportances := Boruta(d, l, 50, 150, maxDepth, numLeaves)
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
	featureStats := REF(train, test, label, tLabel, numIteration, numEstimators, maxDepth, numLeaves)

	fmt.Println("Results:", featureStats)
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
	numIteration := 50
	numEstimators := 100
	// alpha := 0.05
	maxDepth := 0
	numLeaves := 0

	train, label, test, tLabel := SplitTrainTest(dataset, labels, 0.75)
	featuresScore := permutation(train, test, label, tLabel, numIteration, numEstimators, maxDepth, numLeaves)

	fmt.Println("Results:", featuresScore)
}
