package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strconv"

	"github.com/po3rin/gomnist"
)

func PrepareMnistData(num int, numbers []int) (*Dataset, []int) {
	dir := gomnist.NewLoader("./testdata")

	mnist, err := dir.Load()
	if err != nil {
		panic(fmt.Sprintf("Failed to load MNIST dataset: %v", err))
	}

	rows, cols := mnist.TrainData.Dims()

	labelIndices := GetLabelIndices(&mnist, numbers, rows)
	selectedIndices := SampleInstance(num, numbers, labelIndices)
	dataset, label := CreateDataset(&mnist, selectedIndices, cols)

	return dataset, label
}

func GetLabelIndices(mnist *gomnist.MNIST, numbers []int, rows int) map[int][]int {
	labelIndices := make(map[int][]int)

	for i := 0; i < rows; i ++ {
		label := int(mnist.TrainLabels.At(i, 0))

		for _, n := range numbers {
			if label == n {
				labelIndices[label] = append(labelIndices[label], i) 
				break 
			}
		}
	}

	return labelIndices
}

func SampleInstance(num int, numbers []int, labelIndices map[int][]int) []int {
	numLabels := len(numbers)
	numPerLabel := num / numLabels    // floor
	allocation := make(map[int]int)
	total := 0

	for _, label := range numbers {
		capcity := len(labelIndices[label])
		
		// If enough 
		if capcity >= numPerLabel {
			allocation[label] = numPerLabel
			total += numPerLabel
		} else {
			// Not enough
			allocation[label] = capcity
			total += capcity
		}
	}

	remaining := num - total
	for remaining > 0 {
		for _, label := range numbers {

			// Distribute the remaining among others
			if allocation[label] < len(labelIndices[label]) {
				allocation[label] ++
				remaining --
			}

			if remaining == 0{
				break
			}
		}
	}

	selectedIndices := []int{}
	rand := rand.New(rand.NewSource(int64(46)))  	// 36  // 31:18

	for label, count := range allocation {

		indices := labelIndices[label]
		selected := rand.Perm(len(indices))[:count]

		for _, idx := range selected {
			selectedIndices = append(selectedIndices, indices[idx])
		}
	}

	rand.Shuffle(len(selectedIndices), func(i, j int) {
		selectedIndices[i], selectedIndices[j] = selectedIndices[j], selectedIndices[i]
	})

	return selectedIndices
}

func CreateDataset(mnist *gomnist.MNIST, selectedIndices []int, cols int) (*Dataset, []int) {
	dataset := &Dataset{
		Features: make([]string, cols),
		Instance: make([]*Instance, len(selectedIndices)),
		Label: "MNIST",
	}

	labels := make([]int, len(selectedIndices))

	for i := 0; i < cols; i++ {
        dataset.Features[i] = strconv.Itoa(i)  //fmt.Sprintf("pixel%d", i) 
    }

	for i, row := range selectedIndices {
		// make instance 
		instance := &Instance{
			Features: make(map[string]float64, cols),
			Label:  fmt.Sprintf("%d", int(mnist.TrainLabels.At(row, 0))),
		}

		for col := 0; col < cols; col++ {
            instance.Features[dataset.Features[col]] = mnist.TrainData.At(row, col)
        }

		dataset.Instance[i] = instance
		labels[i] = int(mnist.TrainLabels.At(row, 0))
	}

	return dataset, labels
}


func Write2Json(data interface{}, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}

	defer file.Close()

	return json.NewEncoder(file).Encode(data)
}