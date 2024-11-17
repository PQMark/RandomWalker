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

// Fine on toy dataset 
// d does not contain label 
// numIteration: The maximum 
func Boruta(d *Dataset, dLabel []int, numIteration, numEstimators, maxDepth, numLeaves int) ([]string, map[string]int, map[string]float64) {
	removedFeatures := make(map[string]bool)		// features marked as unimportant

	var featuresToConsider []string					// features remians tentative
	
	featureImportances := make(map[string]float64)

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
			trainRandomForestBoruta(d, dLabel, featuresToConsider, numEstimators, maxDepth, numLeaves, results)
		}

		fmt.Println(results)
		threshold := CalculateThreshold(numIteration)
		fmt.Println("Bionomial Threshold:", threshold)

		// Remove unimportant features 
		for f, val := range results {
			if val < threshold {
				
				// record the feature removed 
				removedFeatures[f] = true

				fmt.Println("Delete Feature:", f)

				// remove the feature from featuresToConsider
				featuresToConsider = DeleteFromString(f, featuresToConsider)

			}
		}

		// Converge if there is less than three features or no update any more
		if len(featuresToConsider) < 3 || oldNum == len(featuresToConsider) {
			fmt.Println("Converged.")
			
			// Train a RF with selected features 
			x := ConvertToData(d, featuresToConsider)

			forestWithFeatures := randomforest.Forest{
				Data: randomforest.ForestData{
					X: x,
					Class: dLabel,
				},
			}

			forestWithFeatures.Train(300)

			for i := 0; i < len(featuresToConsider); i ++ {
				featureName := d.Features[i]
				featureImportances[featureName] = forestWithFeatures.FeatureImportance[i]
			}
			
			return featuresToConsider, results, featureImportances
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

// Fine
func trainRandomForestBoruta(d *Dataset, Y []int, features []string, numEstimators, maxDepth, numLeaves int, results map[string]int) {
	
	// Prepare training data and labels for training process
	// convert the data to [][]float64 type 
	x := ConvertToData(d, features)
	//trainY is dLabel

	forest := randomforest.Forest{
		Data: randomforest.ForestData{
			X: x,
			Class: Y,
		},
		MaxDepth: maxDepth,
		LeafSize: numLeaves,
	}

	forest.Train(numEstimators)

	// Find the threshold of shadow features 
	shadow_IS := 0.0
	featuresNum := len(x[0]) / 2 

	for i:=featuresNum; i < 2 * featuresNum; i++ {
		importanceScore := forest.FeatureImportance[i]

		if importanceScore > shadow_IS {
			shadow_IS = importanceScore
		}
	}

	// Update the results 
	// numFeatures := len(x[0])

	for i:=0; i<featuresNum; i++ {
		featureName := d.Features[i]

		if forest.FeatureImportance[i] > shadow_IS {
			results[featureName] ++
		} else if _, exists := results[featureName]; !exists {
            results[featureName] = 0 
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

// num should below 50 to avoid integer overflow 
func CalculateThreshold(num int) int {
	// Null hypothesis: Importance score of a feature exceeds a shadow by random chance --> p_success = 0.5
	p := 0.5

	// set significance to 0.05
	significance := 0.05

	// Initial: 0 success
	current_p := math.Pow(1 - p, float64(num))
	cdf := current_p

	for k := 0; k <= num; k++ {
		
		if 1 - cdf <= significance {
			return k
		}

		// Otherwise keep incrementing k 
		current_p *= (float64(num - k) * p) / (float64(k+1) * (1 - p))
		cdf += current_p
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