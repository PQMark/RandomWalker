package main

import (
	"fmt"
	"os"
	"os/exec"

	randomforest "github.com/malaschitz/randomForest"
)

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
		decayFactor:      1.2,
	})

	modes := []int{50, 100, 150, 300}
	featureImportances := make(map[int]map[string]float64)

	for _, mode := range modes {
		selectedFeatures := getFeaturesRFE(results, mode, 0)

		featureImportances[mode] = make(map[string]float64)

		x := ConvertData(d, selectedFeatures.Features)

		forestWithFeatures := randomforest.Forest{
			Data: randomforest.ForestData{
				X:     x,
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
