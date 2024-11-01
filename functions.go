package main

import (
	"math/rand"
	"fmt"
	"strings"
	"math"
	"github.com/malaschitz/randomForest"
)

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

// d does not contain label 
func Boruta(d, dLabel *Dataset, numIteration, numEstimators int, alpha float64) ([]FeatureImportance, error) {
	removedFeatures := make(map[string]bool)		// features marked as unimportant
	importantFeatures := make(map[string]bool)		// features marked as important
	featuresToConsider := make(map[string]struct{})		// features remians tentative
	featureImportanceScores := make(map[string][]float64)

	// Initialize featuresToConsider to all the features 
	for _, name := range d.Features {
		featuresToConsider[name] = struct{}{}
	}

	// Initialize the dataset with shadow features 
	d.Initialize(featuresToConsider)

	//check the size of training data and labels
	if len(d.Instance) != len(dLabel.Instance) {
		panic("Unequal size of training set and label set")
	} 

	iteration := 0
	for iteration <= numIteration {
		iteration ++
		fmt.Println("Boruta Iteration %d:\n", iteration)

		// Check the features match with shadows
		CheckFeatures(d.Features, featuresToConsider)

		// Shuffle Shadow features
		d.ShuffleShadowFeatures(featuresToConsider)

		// Prepare training data and labels for training process
		// convert the data to [][]float64 type 
		trainX := ConvertToData(d, featuresToConsider)
		trainY := 

		// Train the RF model and extract the importance score 
		// importances: map[string][]float64
		importances, err := d.trainRandomForest(numEstimators)
		if err != nil {
			return nil, err
		}

		for feature, importance := range importances {
			if ! strings.HasPrefix(feature, "shadow") {
				featureImportanceScores[feature] = append(featureImportanceScores[feature], importance)
			}
		}

		// Find out the important, unimportant and tentative features from the featureImportanceScores
		imp, unimp := EvaluateFeatures(featureImportanceScores)

		for _, f := range imp {
			importantFeatures[f] = true
			
			// delete the feature from tentative map 
			delete(featuresToConsider, f)
		}

		for _, f := range unimp {
			removedFeatures[f] = true

			// delete the feature from tentative map 
			delete(featuresToConsider, f)
		}

		// Remove shadow features because we'll have new shawdows added each iteration
		d.removeShadow()

		// Converge if there is no elements in tentative 
		if len(featuresToConsider) == 0 {
			fmt.Println("No more tentative features. Converged.")
			break
		}

	}

	// Rank the features based on importance score 
	featureImportances := Rank(featureImportanceScores)

	return featureImportances, nil
}


func (d *Dataset) Initialize(features map[string]struct{}) {
	for f := range features {
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

func (d *Dataset) ShuffleShadowFeatures(features map[string]struct{}) {

	for f := range features {
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

func (d *Dataset) trainRandomForest(numEstimators int) (map[string]float64, error) {
	instances, err := d.convertToInstances()
	if err != nil {
		return nil, err
	}

	// Use square root of the number features for greater tree diversity; Higher bias for lower variance
	numFeatures := int(math.Sqrt(float64(len(d.Features))))
	rf := ensemble.NewRandomForest(numEstimators, numFeatures)

	// Fit the model 
	err = rf.Fit(instances) 
	if err != nil {
		return nil, err
	}

	// Get feature importance from the model 
	importances := getFeatureImportances(rf)

	return importances, nil
}

func CheckFeatures(allFeatures []string, features map[string]struct{}) {
	allFeaturesSet := make(map[string]struct{})
    
	for _, feature := range allFeatures {
        allFeaturesSet[feature] = struct{}{}
    }

    for feature := range features {
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

func ConvertToData(d *Dataset, featuresToConsider map[string]struct{}) [][]float64 {
	
	data := make([][]float64, len(d.Instance))

	for _, instance := range d.Instance {
		data[i] = make([]float64, 0, 2 * len(featuresToConsider))

		// Append features 
		for _, f := range featuresToConsider {
			data = append(data, d.Instance[f])
		}

		// Append shadows 
		for _, f := range featuresToConsider {
			shadowF := "shadow_" + f
			data = append(data, d.Instance[shadowF])
		}
	}

	return data
}


