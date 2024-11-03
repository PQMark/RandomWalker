package main

import (
	"fmt"
	"math/rand"
)

func main() {
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

    selectedFeatures, featureImportances := Boruta(dataset, labels, numIteration, numEstimators)

	fmt.Println("Selected Features:", selectedFeatures)
    fmt.Println("Feature Importances:", featureImportances)
}