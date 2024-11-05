package main

import (
	"fmt"
	"os/exec"
	"os"
	"math/rand"
)

func main() {
	
	// TestSyntheziedData()

	TestImage()
	
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
        x1 := rand.NormFloat64()       // Important feature
        x2 := rand.NormFloat64()	   // Important feature
        x3 := rand.NormFloat64() 	   // Important feature

        // Redundant features (linear combinations)
        x4 := x1 + x2                  // Redundant feature
        x5 := x2 - x3                  // Redundant feature

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

    selectedFeatures, finalResult, featureImportances := Boruta(dataset, labels, numIteration, numEstimators)

	fmt.Println("Selected Features:", selectedFeatures)
	fmt.Println("Results:", finalResult)
    fmt.Println("Feature Importances:", featureImportances)
}

func TestImage() {
	num := 80
	features := []int {
		1, 2, 
	}
	d, l := PrepareMnistData(num, features)
	fmt.Println(l)

	if err := Write2Json(d, "MNIST.json"); err != nil {
		fmt.Println("Error writing JSON:", err)
	}

	selectedFeatures, finalResult, featureImportances := Boruta(d, l, 50, 150)
	fmt.Println(selectedFeatures)
	fmt.Println(finalResult)
	fmt.Println(featureImportances)

	if err := Write2Json(featureImportances, "MNIST_Output.json"); err != nil {
		fmt.Println("Error writing JSON:", err)
	}

	cmd := exec.Command("python3", "scripts/visualization.py")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Run()

}