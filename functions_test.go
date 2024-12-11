package main

import (
	//"bufio"
	"fmt"
	"math"
	//"os"
	// "strconv"
	// "strings"
	"testing"
	//"path/filepath"
	//"github.com/PQMark/RandomWalker" 
	//"rfe"
)

// type Lr struct {
// 	initialThreshold float64
// 	decayFactor      float64
// }

type FeatureDecayTest struct {
    features  []string            // List of feature strings
    numTotalFeatures int                 // Total number of features
    lrParams         Lr // Learning rate parameters (Alpha, Beta)
    expectedThreshold float64           // Expected output (for comparison)
}

func TestFeatureDecayScheduler(t *testing.T) {
	// Define the test cases
	tests := []struct {
		features       *[]string
		numTotalFeatures int
		lrParams       Lr
		expected       float64
	}{
		{
			features:       &[]string{"feature1", "feature2"},
			numTotalFeatures: 10,
			lrParams:       Lr{initialThreshold: 0.2, decayFactor: 1.5},
			expected:       0.03, // Threshold should be calculated based on the remaining percent
		},
		{
			features:       &[]string{"feature1"},
			numTotalFeatures: 10,
			lrParams:       Lr{initialThreshold: 0.2, decayFactor: 1.5},
			expected:       0.03, // Threshold should be calculated based on the remaining percent
		},
		{
			features:       &[]string{"feature1", "feature2", "feature3", "feature4", "feature5"},
			numTotalFeatures: 10,
			lrParams:       Lr{initialThreshold: 0.0, decayFactor: 1.5}, // Using default initialThreshold of 0.2
			expected:       0.07071, // With 5 features, threshold should be based on the default threshold
		},
		{
			features:       &[]string{"feature1", "feature2", "feature3"},
			numTotalFeatures: 10,
			lrParams:       Lr{initialThreshold: 0.3, decayFactor: 1.5},
			expected:       0.049295, // Threshold should be calculated based on the remaining percent
		},
		{
			features:       &[]string{"feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10"},
			numTotalFeatures: 10,
			lrParams:       Lr{initialThreshold: 0.0, decayFactor: 1.0}, // Using default decayFactor of 1.5
			expected:       0.2, // If there are exactly 10 features, threshold should be the default value
		},
		{
			features:       &[]string{"feature1"},
			numTotalFeatures: 10,
			lrParams:       Lr{initialThreshold: 0.5, decayFactor: 2.0},
			expected:       0.03, // Should calculate based on decayFactor and remainingPercent
		},
		{
			features:       &[]string{"feature1", "feature2"},
			numTotalFeatures: 5,
			lrParams:       Lr{initialThreshold: 0.2, decayFactor: 1.5},
			expected:       0.050596, // With 2 out of 5 features, threshold should be calculated with remainingPercent
		},
		{
			features:       &[]string{},
			numTotalFeatures: 10,
			lrParams:       Lr{initialThreshold: 0.2, decayFactor: 1.5},
			expected:       0.03, // Edge case: No features remaining, threshold should be 0.03
		},
	}

	for i, test := range tests {
		// Run the FeatureDecayScheduler function
		result := FeatureDecayScheduler(test.features, test.numTotalFeatures, test.lrParams)

		// Compare the result with the expected value
		if math.Abs(result-test.expected) > 1e-3 {
			//fmt.Printf("Test %d did not pass\n", i+1)
			t.Errorf("FeatureDecayScheduler(%v, %v, %v) = %v, want %v",
				*test.features, test.numTotalFeatures, test.lrParams, result, test.expected)
		} else {
			fmt.Printf("Test %d Passed!\n", i+1)
		}
	}
}


func TestDiscardFeatures(t *testing.T) {
    tests := []struct {
        data           [][]float64
        features       *[]string
        threshold      float64
        expectedFeatures []string
    }{
        // Case 1: All features are discarded (threshold is high)
        {
            data: [][]float64{
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0},
            },
            features:       &[]string{"feature1", "feature2", "feature3"},
            threshold:      0.9, // Threshold is very high, no features will meet the criteria
            expectedFeatures: []string{}, // Expecting all features to be discarded
        },
        // Case 2: All features are retained (threshold is very low)
        {
            data: [][]float64{
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0},
            },
            features:       &[]string{"feature1", "feature2", "feature3"},
            threshold:      0.01, // Threshold is very low, all features will be retained
            expectedFeatures: []string{"feature1", "feature2", "feature3"}, // Expecting all features to be retained
        },
        // Case 3: Threshold exactly at the point of decision
        {
            data: [][]float64{
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0},
            },
            features:       &[]string{"feature1", "feature2", "feature3"},
            threshold:      0.33, // Threshold is set so that only features above this percentage are retained
            expectedFeatures: []string{"feature2", "feature3"}, // Features retained based on sum comparison
        },
        // Case 4: Single row of data (only one data point)
        {
            data: [][]float64{
                {1.0, 2.0, 3.0},
            },
            features:       &[]string{"feature1", "feature2", "feature3"},
            threshold:      0.5, // Threshold applied to only one data point
            expectedFeatures: []string{"feature2", "feature3"}, // Based on sum comparison for a single row
        },
        // Case 5: Single feature (only one feature in the list)
        {
            data: [][]float64{
                {1.0},
                {2.0},
                {3.0},
            },
            features:       &[]string{"feature1"},
            threshold:      0.5, // Threshold applied to only one feature
            expectedFeatures: []string{"feature1"}, // Only one feature, should be retained
        },
        // Case 6: Empty dataset
        {
            data: [][]float64{}, // No data points
            features:       &[]string{"feature1", "feature2", "feature3"},
            threshold:      0.5, // Threshold applied to an empty dataset
            expectedFeatures: []string{"feature1", "feature2", "feature3"}, // All features should remain intact since there is no data to process
        },
        // Case 7: Empty features list
        {
            data: [][]float64{
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
            },
            features:       &[]string{}, // No features to discard
            threshold:      0.5, // Threshold applied with no features
            expectedFeatures: []string{}, // Expecting no features to be present
        },
    }

    for i, test := range tests {
        // Run the DiscardFeatures function
        DiscardFeatures(test.data, test.features, test.threshold)

        // Check if the resulting features match the expected
        if !equal(test.features, &test.expectedFeatures) {
            fmt.Printf("Test %d did not pass!\n", i+1)
            t.Errorf("Test %d Failed: expected %v, got %v", i+1, test.expectedFeatures, *test.features)
        } else {
            fmt.Printf("Test %d Passed!\n", i+1)
        }
    }
}


// Helper function to compare two slices of strings
func equal(a, b *[]string) bool {
    if len(*a) != len(*b) {
        return false
    }
    for i := range *a {
        if (*a)[i] != (*b)[i] {
            return false
        }
    }
    return true
}



// func DiscardFeatures(data [][]float64, features *[]string, threshold float64) {
// 	length := len(*features)
// 	n := float64(len(data))

// 	size := int(threshold * float64(length))
// 	if size < 1 {
// 		size = 1
// 	}

// 	importanceMean := make([]float64, length)

// 	for c := range data[0] {
// 		sum := 0.0
// 		for r := range data {
// 			sum += data[r][c]
// 		}

// 		importanceMean[c] = sum / n
// 	}

// 	for i := length - 1; i >= 0; i-- {
// 		val1 := importanceMean[i]
// 		count := 0

// 		for _, val2 := range importanceMean {
// 			if val1 > val2 {
// 				count++
// 			}

// 			if count >= size {
// 				break
// 			}
// 		}

// 		if count < size {
// 			*features = DeleteFromString((*features)[i], *features)
// 		}

// 		if len(*features) <= length-size {
// 			break
// 		}

// 	}

// }

// func DeleteFromString(f string, features []string) []string {
// 	// Find the index of f
// 	index := -1
// 	for i, feature := range features {
// 		if feature == f {
// 			index = i
// 			break
// 		}
// 	}

// 	// Panic if f is not found
// 	if index == -1 {
// 		panic(fmt.Sprintf("Cannot delete '%s' since it is not found", f))
// 	} else {
// 		features = append(features[:index], features[index+1:]...)
// 	}

// 	return features
// }

// func FeatureDecayScheduler(features *[]string, numTotalFeatures int, lrParams Lr) float64 {
// 	// default initial threshold and decay factor
// 	if lrParams.initialThreshold == 0.0 {
// 		lrParams.initialThreshold = 0.2
// 	}

// 	if lrParams.decayFactor == 0.0 {
// 		lrParams.decayFactor = 1.5
// 	}

// 	var threshold float64
// 	remainFeatures := len(*features)

// 	//fmt.Println(numTotalFeatures)
// 	remainingPercent := float64(remainFeatures)/float64(numTotalFeatures)
// 	if remainingPercent <= 0.20 {
// 		threshold = 0.03
// 		fmt.Println("t: ", threshold)

// 	} else {
// 		// fmt.Println(initialThreshold)
// 		// fmt.Println(decayFactor)
// 		threshold = float64(lrParams.initialThreshold) * math.Pow(remainingPercent, lrParams.decayFactor)
// 		fmt.Println("t: ", threshold)
// 	}
// 	return threshold
// }