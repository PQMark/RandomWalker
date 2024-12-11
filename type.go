package main

type Dataset struct {
	Instance []*Instance
	Features []string
	Label    string
}

type Instance struct {
	Features map[string]float64
	Label    string
}

type Optimization struct {
	Optimize        bool
	Default         HyperParameters
	DefaultGrid     bool
	HyperParamsGrid []HyperParameters
	numProcs        int
}

type Counts struct {
	TP int
	FP int
	FN int
}

type FeaturesF1 struct {
	FeatureSelected []string
	F1              float64
}

type Numeric interface {
	int | int8 | int16 | int32 | int64 | float32 | float64
}

type HyperParameters struct {
	NTrees   int
	MaxDepth int
	LeafSize int
}

type SearchResult struct {
	Params HyperParameters
	F1Avg  float64
}

type FeatureAvgMean struct {
	Feature          string
	AvgPermutScore   float64
	ErrorPermutScore float64
}

type FeatureStats struct {
	Features []string
	AvgF1    float64
	ErrorF1  float64
}

type Lr struct {
	initialThreshold float64
	decayFactor      float64
}
