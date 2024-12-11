package main

import (
	//"bufio"
	"fmt"
	"math"
	"reflect"

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
	features          []string // List of feature strings
	numTotalFeatures  int      // Total number of features
	lrParams          Lr       // Learning rate parameters (Alpha, Beta)
	expectedThreshold float64  // Expected output (for comparison)
}

func TestFeatureDecayScheduler(t *testing.T) {
	// Define the test cases
	tests := []struct {
		features         *[]string
		numTotalFeatures int
		lrParams         Lr
		expected         float64
	}{
		{
			features:         &[]string{"feature1", "feature2"},
			numTotalFeatures: 10,
			lrParams:         Lr{initialThreshold: 0.2, decayFactor: 1.5},
			expected:         0.03, // Threshold should be calculated based on the remaining percent
		},
		{
			features:         &[]string{"feature1"},
			numTotalFeatures: 10,
			lrParams:         Lr{initialThreshold: 0.2, decayFactor: 1.5},
			expected:         0.03, // Threshold should be calculated based on the remaining percent
		},
		{
			features:         &[]string{"feature1", "feature2", "feature3", "feature4", "feature5"},
			numTotalFeatures: 10,
			lrParams:         Lr{initialThreshold: 0.0, decayFactor: 1.5}, // Using default initialThreshold of 0.2
			expected:         0.07071,                                     // With 5 features, threshold should be based on the default threshold
		},
		{
			features:         &[]string{"feature1", "feature2", "feature3"},
			numTotalFeatures: 10,
			lrParams:         Lr{initialThreshold: 0.3, decayFactor: 1.5},
			expected:         0.049295, // Threshold should be calculated based on the remaining percent
		},
		{
			features:         &[]string{"feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10"},
			numTotalFeatures: 10,
			lrParams:         Lr{initialThreshold: 0.0, decayFactor: 1.0}, // Using default decayFactor of 1.5
			expected:         0.2,                                         // If there are exactly 10 features, threshold should be the default value
		},
		{
			features:         &[]string{"feature1"},
			numTotalFeatures: 10,
			lrParams:         Lr{initialThreshold: 0.5, decayFactor: 2.0},
			expected:         0.03, // Should calculate based on decayFactor and remainingPercent
		},
		{
			features:         &[]string{"feature1", "feature2"},
			numTotalFeatures: 5,
			lrParams:         Lr{initialThreshold: 0.2, decayFactor: 1.5},
			expected:         0.050596, // With 2 out of 5 features, threshold should be calculated with remainingPercent
		},
		{
			features:         &[]string{},
			numTotalFeatures: 10,
			lrParams:         Lr{initialThreshold: 0.2, decayFactor: 1.5},
			expected:         0.03, // Edge case: No features remaining, threshold should be 0.03
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
		data             [][]float64
		features         *[]string
		threshold        float64
		expectedFeatures []string
	}{
		// Case 1: Deleting multiple features when the threshold is high
		{
			data: [][]float64{
				{1.0, 2.0, 3.0, 4.0},
				{4.0, 5.0, 6.0, 7.0},
				{7.0, 8.0, 9.0, 9.5},
			},
			features:         &[]string{"feature1", "feature2", "feature3", "feature4"},
			threshold:        0.5,                              // Threshold is high, likely discarding features
			expectedFeatures: []string{"feature3", "feature4"}, // Retained features are expected based on their importance
		},

		// Case 2: High threshold resulting in the deletion of most features
		{
			data: [][]float64{
				{1.0, 2.0, 3.0},
				{4.0, 5.0, 6.0},
				{7.0, 8.0, 9.0},
			},
			features:         &[]string{"feature1", "feature2", "feature3"},
			threshold:        0.9,                  // High threshold, expect to retain only the most important feature
			expectedFeatures: []string{"feature3"}, // Only one feature retained
		},

		// Case 3: Low threshold resulting in retaining most features
		{
			data: [][]float64{
				{1.0, 2.0, 3.0},
				{4.0, 5.0, 6.0},
				{7.0, 8.0, 9.0},
			},
			features:         &[]string{"feature1", "feature2", "feature3"},
			threshold:        0.01,                             // Very low threshold, expect all features to be retained
			expectedFeatures: []string{"feature2", "feature3"}, // at least one feature removed
		},

		// Case 4: Threshold is exactly at the point of decision
		{
			data: [][]float64{
				{1.0, 2.0, 3.0},
				{4.0, 5.0, 6.0},
				{7.0, 8.0, 9.0},
			},
			features:         &[]string{"feature1", "feature2", "feature3"},
			threshold:        0.33,                             // Features that are above this threshold should be retained
			expectedFeatures: []string{"feature2", "feature3"}, // Features based on threshold comparison
		},

		// Case 5: Single data point (row), apply threshold to one feature
		{
			data: [][]float64{
				{1.0, 2.0, 3.0},
			},
			features:         &[]string{"feature1", "feature2", "feature3"},
			threshold:        0.5,                              // Only one feature with a threshold applied
			expectedFeatures: []string{"feature2", "feature3"}, // Expected features after threshold application
		},

		// Case 6: Single feature (only one feature in the list)
		{
			data: [][]float64{
				{1.0},
				{2.0},
				{3.0},
			},
			features:         &[]string{"feature1"},
			threshold:        0.5,        // Threshold applied to a single feature
			expectedFeatures: []string{}, // No feature retained as it is deleted
		},

		// Case 7: Empty dataset (no data points)
		{
			data:             [][]float64{}, // No data points
			features:         &[]string{"feature1", "feature2", "feature3"},
			threshold:        0.5,                                          // Threshold applied to an empty dataset
			expectedFeatures: []string{"feature1", "feature2", "feature3"}, // No features discarded
		},

		// Case 8: Empty features list (no features to discard)
		{
			data: [][]float64{
				{1.0, 2.0, 3.0},
				{4.0, 5.0, 6.0},
			},
			features:         &[]string{}, // No features
			threshold:        0.5,         // Threshold applied with no features
			expectedFeatures: []string{},  // No features to retain or discard
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

func TestNormalization(t *testing.T) {
	tests := []struct {
		data         []float64
		expectedData []float64
	}{
		// Case 1: Normal case with positive numbers
		{
			data:         []float64{1.0, 2.0, 3.0, 4.0},
			expectedData: []float64{0.1, 0.2, 0.3, 0.4}, // Normalized values, each value divided by 10
		},

		// Case 2: Normal case with decimal values
		{
			data:         []float64{0.1, 0.2, 0.3},
			expectedData: []float64{0.1 / 0.6, 0.2 / 0.6, 0.3 / 0.6}, // Sum is 0.6
		},

		// Case 3: All elements are zero
		{
			data:         []float64{0.0, 0.0, 0.0},
			expectedData: []float64{0.0, 0.0, 0.0}, // Sum is 0, and dividing by 0 should leave all elements as 0
		},

		// Case 4: Single element
		{
			data:         []float64{10.0},
			expectedData: []float64{1.0}, // Only one element, and it becomes 1 after division by itself
		},

		// Case 5: Large numbers
		{
			data:         []float64{1000.0, 2000.0, 3000.0},
			expectedData: []float64{0.1667, 0.3333, 0.5}, // Sum is 6000, and values divided by 6000
		},

		// Case 6: Negative numbers
		{
			data:         []float64{-1.0, -2.0, -3.0},
			expectedData: []float64{0.1667, 0.3333, 0.5}, // Sum is -6, and values divided by -6
		},

		// Case 7: Empty array
		{
			data:         []float64{},
			expectedData: []float64{}, // No elements to normalize
		},

		// Case 8: All elements are the same
		{
			data:         []float64{5.0, 5.0, 5.0},
			expectedData: []float64{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}, // Sum is 15, each element divided by 15
		},
	}

	for i, test := range tests {
		// Run the Normalization function
		Normalization(test.data)

		// Check if the resulting data matches the expected
		if !equalFloat64(test.data, test.expectedData) {
			fmt.Printf("Test %d did not pass!\n", i+1)
			t.Errorf("Test %d Failed: expected %v, got %v", i+1, test.expectedData, test.data)
		} else {
			fmt.Printf("Test %d Passed!\n", i+1)
		}
	}
}

func TestCheckIfAllNegative(t *testing.T) {
	tests := []struct {
		name string
		data []float64
		want bool
	}{
		{
			name: "All negative values",
			data: []float64{-1.0, -2.0, -3.0},
			want: true,
		},
		{
			name: "All values are zero",
			data: []float64{0.0, 0.0, 0.0},
			want: true,
		},
		{
			name: "All positive values",
			data: []float64{1.0, 2.0, 3.0},
			want: false,
		},
		{
			name: "Some positive and some negative values",
			data: []float64{-1.0, 2.0, -3.0},
			want: false,
		},
		{
			name: "Empty data",
			data: []float64{},
			want: true,
		},
		{
			name: "Positive and zero values",
			data: []float64{0.0, 1.0, -2.0},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CheckIfAllNegative(tt.data)
			if got != tt.want {
				t.Errorf("CheckIfAllNegative(%v) = %v, want %v", tt.data, got, tt.want)
			}
		})
	}
}

func TestGetMaxIndexInt(t *testing.T) {
	tests := []struct {
		name string
		data []int
		want int
	}{
		{
			name: "Positive integers",
			data: []int{4, 1, 7, 2, 5},
			want: 2,
		},
		{
			name: "Negative integers",
			data: []int{-4, -1, -7, -2, -5},
			want: 1,
		},
		{
			name: "Mixed positive and negative integers",
			data: []int{-4, 1, -7, 2, 5},
			want: 4,
		},
		{
			name: "All elements are the same",
			data: []int{3, 3, 3, 3, 3},
			want: 0,
		},
		{
			name: "Single element",
			data: []int{42},
			want: 0,
		},
		{
			name: "Empty list",
			data: []int{},
			want: -1,
		},
		{
			name: "Large numbers",
			data: []int{1000000000, 999999999, 987654321, 1000000001},
			want: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getMaxIndex(tt.data)
			if got != tt.want {
				t.Errorf("getMaxIndex(%v) = %d, want %d", tt.data, got, tt.want)
			}
		})
	}
}

func TestGetMaxIndexFloat(t *testing.T) {
	tests := []struct {
		name string
		data []float64
		want int
	}{
		{
			name: "Positive floats",
			data: []float64{4.5, 1.1, 7.7, 2.2, 5.5},
			want: 2,
		},
		{
			name: "Negative floats",
			data: []float64{-4.5, -1.1, -7.7, -2.2, -5.5},
			want: 1,
		},
		{
			name: "Mixed positive and negative floats",
			data: []float64{-4.5, 1.1, -7.7, 2.2, 5.5},
			want: 4,
		},
		{
			name: "All elements are the same",
			data: []float64{3.3, 3.3, 3.3, 3.3, 3.3},
			want: 0,
		},
		{
			name: "Single element",
			data: []float64{42.42},
			want: 0,
		},
		{
			name: "Empty list",
			data: []float64{},
			want: -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getMaxIndex(tt.data)
			if got != tt.want {
				t.Errorf("getMaxIndex(%v) = %d, want %d", tt.data, got, tt.want)
			}
		})
	}
}

func TestDelete(t *testing.T) {
	tests := []struct {
		name string
		data []float64
		idx  int
		want []float64
	}{
		{
			name: "Delete middle element",
			data: []float64{1.1, 2.2, 3.3, 4.4},
			idx:  2,
			want: []float64{1.1, 2.2, 4.4},
		},
		{
			name: "Delete first element",
			data: []float64{5.5, 6.6, 7.7, 8.8},
			idx:  0,
			want: []float64{6.6, 7.7, 8.8},
		},
		{
			name: "Delete last element",
			data: []float64{10.0, 20.0, 30.0, 40.0},
			idx:  3,
			want: []float64{10.0, 20.0, 30.0},
		},
		{
			name: "Index out of bounds",
			data: []float64{100.0, 200.0, 300.0},
			idx:  -1,
			want: []float64{100.0, 200.0, 300.0},
		},
		{
			name: "Index out of bounds",
			data: []float64{100.0, 200.0, 300.0},
			idx:  4,
			want: []float64{100.0, 200.0, 300.0},
		},
		{
			name: "Empty slice",
			data: []float64{},
			idx:  0,
			want: []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Delete(tt.data, tt.idx)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Delete(%v, %d) = %v, want %v", tt.data, tt.idx, got, tt.want)
			}
		})
	}
}

func TestPairwiseDeductionFloat(t *testing.T) {
	tests := []struct {
		name  string
		data1 []float64
		data2 []float64
		want  []float64
	}{
		{
			name:  "Equal length positive floats",
			data1: []float64{10.5, 20.5, 30.5, 40.5},
			data2: []float64{1.5, 2.5, 3.5, 4.5},
			want:  []float64{9.0, 18.0, 27.0, 36.0},
		},
		{
			name:  "Equal length mixed floats",
			data1: []float64{5.5, -2.5, 7.0, -3.5},
			data2: []float64{3.0, 1.5, -5.5, -2.5},
			want:  []float64{2.5, -4.0, 12.5, -1.0},
		},
		{
			name:  "Empty slices",
			data1: []float64{},
			data2: []float64{},
			want:  []float64{},
		},
		{
			name:  "Single element slices",
			data1: []float64{42.42},
			data2: []float64{24.24},
			want:  []float64{18.18},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PairwiseDeduction(tt.data1, tt.data2)
			if !equalFloat64(got, tt.want) {
				t.Errorf("PairwiseDeduction(%v, %v) = %v, want %v", tt.data1, tt.data2, got, tt.want)
			}
		})
	}
}

func TestPairwiseDeductionPanic(t *testing.T) {
	tests := []struct {
		name  string
		data1 []int
		data2 []int
	}{
		{
			name:  "Unequal length slices",
			data1: []int{10, 20, 30},
			data2: []int{1, 2},
		},
		{
			name:  "One slice empty",
			data1: []int{1, 2, 3},
			data2: []int{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("Expected panic for input %v and %v", tt.data1, tt.data2)
				}
			}()
			_ = PairwiseDeduction(tt.data1, tt.data2)
		})
	}
}

// Helper function to compare slices of float64 with a small epsilon for floating point precision
func equalFloat64(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-3 { // small epsilon tolerance for floating-point comparison
			return false
		}
	}
	return true
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
