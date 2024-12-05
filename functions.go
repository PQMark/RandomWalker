package main

import (
	//"fmt"
	"math/rand"
	"math"

	"github.com/malaschitz/randomForest"
)

type Counts struct{
	TP int 
	FP int
	FN int 
}

// Train + Test
func SplitTrainTest(data *Dataset, labels []int, ratio float64) (*Dataset, []int, *Dataset, []int) {
	if ratio < 0 || ratio >= 1 {
		panic("Ratio should be within 0 and 1")
	}

	if len(data.Instance) != len(labels) {
		panic("Wrong number of instances and labels")
	}

	indices := rand.Perm(len(labels))
	split := int(ratio * float64(len(labels)))

	trainData := &Dataset{
		Instance: make([]*Instance, 0, split),
		Features: data.Features,
		Label: data.Label,
	}
	testData := &Dataset{
		Instance: make([]*Instance, 0, len(labels) - split),
		Features: data.Features,
		Label: data.Label,
	}
	trainlabelss := make([]int, 0, split)
	testlabelss := make([]int, 0, len(labels) - split)

	for i, idx := range indices {
		if i < split {
			trainData.Instance = append(trainData.Instance, data.Instance[idx])
			trainlabelss = append(trainlabelss, labels[idx])
		} else {
			testData.Instance = append(testData.Instance, data.Instance[idx])
			testlabelss = append(testlabelss, labels[idx])
		}
	}

	return trainData, trainlabelss, testData, testlabelss
}

// Folds of data 
// Fine 
func FoldSplit(data *Dataset, labels []int, numFolds int) ([]*Dataset, [][]int){
	if numFolds < 2{
		panic("At least 2 folds")
	}

	if len(data.Instance) != len(labels) {
		panic("Wrong number of instances and labelss")
	}

	// shuffle the data 
	indices := rand.Perm(len(labels))
	foldSize := len(indices) / numFolds
	remainder := len(indices) % numFolds

	// Distribute the remainder
	foldSizes := make([]int, numFolds)
	for i := range(foldSizes) {
		foldSizes[i] = foldSize
		if i < remainder {
			foldSizes[i] ++
		}
	}

	dataFolds := make([]*Dataset, numFolds)
	labelFolds := make([][]int, numFolds)
	
	offset := 0
	for i, size := range foldSizes {
		
		dataset := &Dataset{
			Instance: make([]*Instance, 0, size),
			Features: data.Features,
			Label: data.Label,
		}
		label := make([]int, 0, size)

		for n := 0; n < size; n++ {
			idx := indices[n + offset]
			dataset.Instance = append(dataset.Instance, data.Instance[idx])
			label = append(label, labels[idx])
		}

		dataFolds[i] = dataset
		labelFolds[i] = label

		offset += size
	}

	return dataFolds, labelFolds
}

// Fine 
func GetFoldData(dataFolds []*Dataset, labelFolds [][]int, i int) (*Dataset, []int, *Dataset, []int) {
	numFolds := len(labelFolds)

	valData := dataFolds[i]
	valLabel := labelFolds[i]

	trainSize := 0
    for n := 0; n < numFolds; n++ {
        if n != i {
            trainSize += len(labelFolds[n])
        }
    }

	trainData := &Dataset{
		Instance: make([]*Instance, 0, trainSize),
		Features: dataFolds[0].Features,
		Label: dataFolds[0].Label,
	}
	trainLabels := make([]int, 0)

	for n := 0; n < numFolds; n++ {
		if n != i {
			trainData.Instance = append(trainData.Instance, dataFolds[n].Instance...)
			trainLabels = append(trainLabels, labelFolds[n]...)
		}
	}

	return trainData, trainLabels, valData, valLabel
}

// Based on weightedVotes 
// Fine
func Predict(forest *randomforest.Forest, data [][]float64) []int {
	predictions := make([]int, len(data))

	for i, x := range data {
		votes := forest.WeightVote(x)

		maxClass := 0
		maxVote := votes[0]

		for j := 1; j < len(votes); j ++ {
			if votes[j] > maxVote {
				maxVote = votes[j]
				maxClass = j
			}
		}

		predictions[i] = maxClass
	}

	return predictions
}

// Weighted F1 score 
// Fine
func GetF1Score(prediction []int, real []int) float64 {
	if len(prediction) != len(real) {
		panic("predictions and true labels must have the same length")
	}

	counts := make(map[int]*Counts)

	for i := 0; i < len(prediction); i++ {
		pred := prediction[i]
		label := real[i]

		// Initialize
		if counts[pred] == nil {
			counts[pred] = &Counts{}
		}
		if counts[label] == nil {
			counts[label] = &Counts{}
		}

		if pred == label {
			counts[pred].TP ++
		} else {
			counts[pred].FP ++
			counts[label].FN++
		}
	}

	var weightedF1 float64
	totalSupport := 0

	for _, cnt := range counts {
		var precision, recall, f1 float64

		// Calculate precision 
		if cnt.TP + cnt.FP == 0 {
			precision = 0.0
		} else {
			precision = float64(cnt.TP) / float64(cnt.TP + cnt.FP)
		}

		// Calculate recall 
		if cnt.TP + cnt.FN == 0{
			recall = 0.0
		} else {
			recall =  float64(cnt.TP) / float64(cnt.TP + cnt.FN)
		}

		// Calculate F1 score 
		if precision + recall == 0{
			f1 = 0.0
		} else {
			f1 = 2.0 * (precision * recall) / (precision + recall)
		}

		weightedF1 += f1 * float64(cnt.FN + cnt.TP)
		totalSupport += cnt.FN + cnt.TP

		// fmt.Printf("Class %d: Precision=%.4f, Recall=%.4f, F1=%.4f", cls, precision, recall, f1)
	}

	weightedF1 /= float64(totalSupport)

	return weightedF1
}	

func Average(lst []float64) float64 {
	
	a := 0.0

	for _, val := range lst {
		a += val 
	}

	return a / float64(len(lst))
}

func standardError(data []float64, mean float64) float64 {
	n := len(data)
	if n == 0 {
		return 0.0
	}

	// Calculate SD 
	sumSquaredDiffs := 0.0
	for _, val := range data {
		diff := val - mean
		sumSquaredDiffs += diff * diff
	}
	std := math.Sqrt(sumSquaredDiffs / float64(n-1))

	return std / math.Sqrt(float64(n))
}
