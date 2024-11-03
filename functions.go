package main

import (
	"math/rand"
	"fmt"
	"math"
	"github.com/malaschitz/randomForest"
)
//
type Dataset struct {
	Instance []*Instance
	Features []string
	Label string
}

type Instance struct {
	Features map[string]float64
	Label string
}

type FeatureImportance struct {
	Feature string
	ImportanceMean float64
	ImportanceStd float64 
}

// Not tested
// d does not contain label 
// broken
// numIteration: The maximum 
func Boruta(d *Dataset, dLabel []int, numIteration, numEstimators int, alpha float64) ([]string, map[string]int) {
	removedFeatures := make(map[string]bool)		// features marked as unimportant

	var featuresToConsider []string					// features remians tentative
	
	// featureImportanceScores := make(map[string][]float64)

	// Initialize featuresToConsider to all the features 
	for _, name := range d.Features {
		featuresToConsider = append(featuresToConsider, name)
	}

	//check the size of training data and labels
	if len(d.Instance) != len(dLabel) {
		panic("Unequal size of training set and label set")
	} 

	run := 0
	for {
		run ++

		oldNum := len(featuresToConsider)

		// Make a copy of the data with features to consider
		d := DeepCopy(d, featuresToConsider)

		// Initialize the dataset with shadow features 
		d.Initialize(featuresToConsider)

		// Check the features match with shadows
		CheckFeatures(d.Features, featuresToConsider)

		results := make(map[string]int)

		// Train the RF model numIteration times
		for i:=0; i<numIteration; i++ {
			fmt.Println("Boruta run:", run, "/", i)
			
			// Shuffle Shadow features
			d.ShuffleShadowFeatures(featuresToConsider)

			// Train the model and update the results 
			trainRandomForest(d, dLabel, featuresToConsider, numEstimators, results)
		}

		threshold := CalculateThreshold(numIteration)

		// Remove unimportant features 
		for f, val := range results {
			if val < threshold {
				
				// record the feature removed 
				removedFeatures[f] = true

				// remove the feature from featuresToConsider
				DeleteFromString(f, featuresToConsider)

			}
		}

		// Converge if there is less than three features or no update any more
		if len(featuresToConsider) < 3 || oldNum == len(featuresToConsider) {
			fmt.Println("Converged.")
			
			return featuresToConsider, results
		}

	}

}

// fine
func (d *Dataset) Initialize(features []string) {
	for _, f := range features {
		f_shadow := "shadow_" + f 
		d.Features = append(d.Features, f_shadow)

		// values stores the values of a feature 
		var values []float64

		// Extract the values for the feature
		for _, instance := range d.Instance {
			if val, ok := instance.Features[f]; ok {
				values = append(values, val)
			}
		}

		// Add the shuffled shadow 
		for i, instance := range d.Instance {
			instance.Features[f_shadow] = values[i]
		}
	}
}

// fine 
func (d *Dataset) ShuffleShadowFeatures(features []string) {

	for _, f := range features {
		f_shadow := "shadow_" + f 

		// values stores the values of a feature 
		var values []float64

		// Extract the values for the feature
		for _, instance := range d.Instance {
			if val, ok := instance.Features[f_shadow]; ok {
				values = append(values, val)
			}
		}

		// Shuffle the values for shadow
		rand.Shuffle(len(values), func (i, j int)  {
			values[i], values[j] = values[j], values[i]
		})

		// Add the shuffled shadow 
		for i, instance := range d.Instance {
			instance.Features[f_shadow] = values[i]
		}
	}
}

// Not tested 
func trainRandomForest(d *Dataset, Y []int, features []string, numEstimators int, results map[string]int) {
	
	// Prepare training data and labels for training process
	// convert the data to [][]float64 type 
	x := ConvertToData(d, features)
	//trainY is dLabel

	forest := randomforest.Forest{
		Data: randomforest.ForestData{
			X: x,
			Class: Y,
		},
	}

	forest.Train(numEstimators)

	// Find the threshold of shadow features 
	shadow_IS := 0.0
	featuresNum := len(x[0])

	for i:=featuresNum; i < 2 * featuresNum; i++ {
		importanceScore := forest.FeatureImportance[i]

		if importanceScore > shadow_IS {
			shadow_IS = importanceScore
		}
	}
	
	// Update the results 
	numFeatures := len(x[0])

	for i:=0; i<numFeatures; i++ {
		if forest.FeatureImportance[i] > shadow_IS {

			featureName := d.Features[i]
			results[featureName] ++

		}
	}
}


// fine 
func CheckFeatures(allFeatures []string, features []string) {
	allFeaturesSet := make(map[string]struct{})
    
	for _, feature := range allFeatures {
        allFeaturesSet[feature] = struct{}{}
    }

    for _, feature := range features {
        // Check if feature is in allFeaturesSet
        if _, exists := allFeaturesSet[feature]; !exists {
            panic(fmt.Sprintf("Feature '%s' is not in allFeatures", feature))
        }

        // Check if "shadow_" + feature is in allFeaturesSet
        shadowFeature := "shadow_" + feature
        if _, exists := allFeaturesSet[shadowFeature]; !exists {
            panic(fmt.Sprintf("Shadow feature '%s' is not in allFeatures", shadowFeature))
        }
    }

}

// fine 
func ConvertToData(d *Dataset, featuresToConsider []string) [][]float64 {
	
	data := make([][]float64, len(d.Instance))

	for i, instance := range d.Instance {
		data[i] = make([]float64, 0, 2 * len(featuresToConsider))

		// Append features 
		for _, f := range featuresToConsider {
			data[i] = append(data[i], instance.Features[f])
		}

		// Append shadows 
		for _, f := range featuresToConsider {
			shadowF := "shadow_" + f
			data[i] = append(data[i], instance.Features[shadowF])
		}
	}

	return data
}

// not tested 
func CalculateThreshold(num int) int {
	// Null hypothesis: Importance score of a feature exceeds a shadow by random chance --> p_success = 0.5
	p := 0.5

	// set significance to 0.05
	significance := 0.05

	// Initial: 0 success 
	cp := math.Pow(1 - p, float64(num))

	for k := 0; k <= num; k++ {
		
		if 1 - cp <= significance {
			return k
		}

		// Otherwise keep incrementing k 
		cp *= (float64(num - k) * p) / (float64(k+1) * (1 - p))
	}

	return num
}

// fine
func DeepCopy(d *Dataset, features []string) *Dataset {
	copyDataset := &Dataset{
		Features: features,
		Label: d.Label,
	}

	for _, instance := range d.Instance {
		copyInstance := &Instance{
			Features: make(map[string]float64),
			Label: instance.Label,
		}

		// Iterate over and append the selected features
		for _, feature := range features {
			if val, exist := instance.Features[feature]; exist {
				copyInstance.Features[feature] = val
			}
		}

		// Add the new instance to the copied dataset 
		copyDataset.Instance = append(copyDataset.Instance, copyInstance)
	}

	return copyDataset
}

// fine
func DeleteFromString(f string, features []string) []string {
	// Find the index of f 
	index := -1 
	for i, feature := range features {
		if feature == f {
			index = i 
			break
		}
	}

	// Panic if f is not found 
	if index == -1 {
		panic(fmt.Sprintf("Cannot delete '%s' since it is not found", f))
	} else {
		features = append(features[:index], features[index+1: ]...)
	}
	
	return features
}