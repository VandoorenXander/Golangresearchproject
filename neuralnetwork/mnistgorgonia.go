package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"time"
	"io"
	"path/filepath"
	"runtime/pprof"
	"syscall"
	"github.com/pkg/errors"
	"gopkg.in/cheggaaa/pb.v1"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	epochs     = flag.Int("epochs", 10, "Number of epochs to train for")
	dataset    = flag.String("dataset", "test", "dataset to train train or test")
	dtype      = flag.String("dtype", "float64", "Which dtype to use")
	batchsize  = flag.Int("batchsize", 128, "Batch size")
	cpuprofile = flag.String("cpuprofile", "", "CPU profiling")
)

const loc = "./datamnist/"

type RawImage []byte
type Label uint8

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
	Width = 28
	Height = 28
)
const numLabels = 10
const pixelRange = 255

var dt tensor.Dtype

func main() {
	flag.Parse()
	parseDtype()
	rand.Seed(1000)
	log.Printf("gorgonia. %t", gorgonia.CUDA)
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	doneChan := make(chan bool, 1)
	var inputs, targets tensor.Tensor
	var err error
	for _, typ := range []string{"test", "train"} {
		inputs, targets, err := Load(typ, loc, tensor.Float64)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(typ+" inputs:", inputs.Shape())
		fmt.Println(typ+" data:", targets.Shape())
	}
	trainOn := *dataset
	if inputs, targets, err = Load(trainOn, loc, dt); err != nil {
		log.Fatal(err)
	}
	numExamples := inputs.Shape()[0]
	bs := *batchsize
	if err := inputs.Reshape(numExamples, 1, 28, 28); err != nil {
		log.Fatal(err)
	}
	g := gorgonia.NewGraph()
	x := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(bs, 1, 28, 28), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 10), gorgonia.WithName("y"))
	m := newConvNet(g)
	if err = m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}
	
	losses := gorgonia.Must(gorgonia.Log(gorgonia.Must(gorgonia.HadamardProd(m.out, y))))
	cost := gorgonia.Must(gorgonia.Mean(losses))
	cost = gorgonia.Must(gorgonia.Neg(cost))
	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)

	if _, err = gorgonia.Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}
	prog, locMap, _ := gorgonia.Compile(g)

	vm := gorgonia.NewTapeMachine(g, gorgonia.WithPrecompiled(prog, locMap), gorgonia.BindDualValues(m.learnables()...))
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(bs)))
	defer vm.Close()
	var profiling bool
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		profiling = true
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	go cleanup(sigChan, doneChan, profiling)

	batches := numExamples / bs
	log.Printf("Batches %d", batches)
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second)
	bar.SetMaxWidth(80)

	for i := 0; i < *epochs; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()
		for b := 0; b < batches; b++ {
			start := b * bs
			end := start + bs
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			var xVal, yVal tensor.Tensor
			if xVal, err = inputs.Slice(gorgonia.S(start, end)); err != nil {
				log.Fatal("Unable to slice x")
			}

			if yVal, err = targets.Slice(gorgonia.S(start, end)); err != nil {
				log.Fatal("Unable to slice y")
			}
			if err = xVal.(*tensor.Dense).Reshape(bs, 1, 28, 28); err != nil {
				log.Fatalf("Unable to reshape %v", err)
			}

			gorgonia.Let(x, xVal)
			gorgonia.Let(y, yVal)
			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d, batch %d. Error: %v", i, b, err)
			}
			if err = solver.Step(gorgonia.NodesToValueGrads(m.learnables())); err != nil {
				log.Fatalf("Failed to update nodes with gradients at epoch %d, batch %d. Error %v", i, b, err)
			}
			vm.Reset()
			bar.Increment()
		}
		log.Printf("Epoch %d | loss %v", i, costVal)

	}
}
func cleanup(sigChan chan os.Signal, doneChan chan bool, profiling bool) {
	select {
	case <-sigChan:
		log.Println("EMERGENCY EXIT!")
		if profiling {
			log.Println("Stop profiling")
			pprof.StopCPUProfile()
		}
		os.Exit(1)

	case <-doneChan:
		return
	}
}

func handlePprof(sigChan chan os.Signal, doneChan chan bool) {
	var profiling bool
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		profiling = true
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	go cleanup(sigChan, doneChan, profiling)
}

//cnn
type convnet struct {
	g                  *gorgonia.ExprGraph
	w0, w1, w2, w3, w4 *gorgonia.Node 
	d0, d1, d2, d3     float64        

	out *gorgonia.Node
}

func newConvNet(g *gorgonia.ExprGraph) *convnet {
	w0 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(32, 1, 3, 3), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w1 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(64, 32, 3, 3), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w2 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(128, 64, 3, 3), gorgonia.WithName("w2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w3 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(128*3*3, 625), gorgonia.WithName("w3"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w4 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(625, 10), gorgonia.WithName("w4"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	return &convnet{
		g:  g,
		w0: w0,
		w1: w1,
		w2: w2,
		w3: w3,
		w4: w4,

		d0: 0.2,
		d1: 0.2,
		d2: 0.2,
		d3: 0.2,
	}
}
func (m *convnet) learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.w1, m.w2, m.w3, m.w4}
}
func (m *convnet) fwd(x *gorgonia.Node) (err error) {
	var c0, c1, c2, fc *gorgonia.Node
	var a0, a1, a2, a3 *gorgonia.Node
	var p0, p1, p2 *gorgonia.Node
	var l0, l1, l2, l3 *gorgonia.Node
	// LAYER 0
	if c0, err = gorgonia.Conv2d(x, m.w0, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 0 Convolution failed")
	}
	if a0, err = gorgonia.Rectify(c0); err != nil {
		return errors.Wrap(err, "Layer 0 activation failed")
	}
	if p0, err = gorgonia.MaxPool2D(a0, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "Layer 0 Maxpooling failed")
	}
	log.Printf("p0 shape %v", p0.Shape())
	if l0, err = gorgonia.Dropout(p0, m.d0); err != nil {
		return errors.Wrap(err, "Unable to apply a dropout")
	}
	// Layer 1
	if c1, err = gorgonia.Conv2d(l0, m.w1, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 1 Convolution failed")
	}
	if a1, err = gorgonia.Rectify(c1); err != nil {
		return errors.Wrap(err, "Layer 1 activation failed")
	}
	if p1, err = gorgonia.MaxPool2D(a1, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "Layer 1 Maxpooling failed")
	}
	if l1, err = gorgonia.Dropout(p1, m.d1); err != nil {
		return errors.Wrap(err, "Unable to apply a dropout to layer 1")
	}

	// Layer 2
	if c2, err = gorgonia.Conv2d(l1, m.w2, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 2 Convolution failed")
	}
	if a2, err = gorgonia.Rectify(c2); err != nil {
		return errors.Wrap(err, "Layer 2 activation failed")
	}
	if p2, err = gorgonia.MaxPool2D(a2, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "Layer 2 Maxpooling failed")
	}
	if l2, err = gorgonia.Dropout(p2, m.d2); err != nil {
		return errors.Wrap(err, "Unable to apply a dropout to layer 1")
	}

	var r2 *gorgonia.Node
	b, c, h, w := p2.Shape()[0], p2.Shape()[1], p2.Shape()[2], p2.Shape()[3]
	if r2, err = gorgonia.Reshape(p2, tensor.Shape{b, c * h * w}); err != nil {
		return errors.Wrap(err, "Unable to reshape layer 2")
	}
	log.Printf("r2 shape %v", r2.Shape())
	if l2, err = gorgonia.Dropout(r2, m.d2); err != nil {
		return errors.Wrap(err, "Unable to apply a dropout on layer 2")
	}

	// Layer 3
	if fc, err = gorgonia.Mul(l2, m.w3); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w3")
	}
	if a3, err = gorgonia.Rectify(fc); err != nil {
		return errors.Wrapf(err, "Unable to activate fc")
	}
	if l3, err = gorgonia.Dropout(a3, m.d3); err != nil {
		return errors.Wrapf(err, "Unable to apply a dropout on layer 3")
	}

	// output decode
	var out *gorgonia.Node
	if out, err = gorgonia.Mul(l3, m.w4); err != nil {
		return errors.Wrapf(err, "Unable to multiply l3 and w4")
	}
	m.out, err = gorgonia.SoftMax(out)
	return
}

func parseDtype() {
	switch *dtype {
	case "float64":
		dt = tensor.Float64
	case "float32":
		dt = tensor.Float32
	default:
		log.Fatalf("Unknown dtype: %v", *dtype)
	}
}

//inlezen data

func Load(typ, loc string, as tensor.Dtype) (inputs, targets tensor.Tensor, err error) {
	const (
		trainLabel = "train-labels.idx1-ubyte"
		trainData  = "train-images.idx3-ubyte"
		testLabel  = "t10k-labels.idx1-ubyte"
		testData   = "t10k-images.idx3-ubyte"
	)
	var labelFile, dataFile string
	switch typ {
	case "train", "dev":
		labelFile = filepath.Join(loc, trainLabel)
		dataFile = filepath.Join(loc, trainData)
	case "test":
		labelFile = filepath.Join(loc, testLabel)
		dataFile = filepath.Join(loc, testData)
	}
	var labelData []Label
	var imageData []RawImage

	if labelData, err = readLabelFile(os.Open(labelFile)); err != nil {
		return nil, nil, errors.Wrap(err, "Unable to read Labels")
	}

	if imageData, err = readImageFile(os.Open(dataFile)); err != nil {
		return nil, nil, errors.Wrap(err, "Unable to read image data")
	}

	inputs = prepareX(imageData, as)
	targets = prepareY(labelData, as)
	return
}
func pixelWeight(px byte) float64 {
	retVal := float64(px)/pixelRange*0.9 + 0.1
	if retVal == 1.0 {
		return 0.999
	}
	return retVal
}

func reversePixelWeight(px float64) byte {
	return byte((pixelRange*px - pixelRange) / 0.9)
}

func prepareX(M []RawImage, dt tensor.Dtype) (retVal tensor.Tensor) {
	rows := len(M)
	cols := len(M[0])

	var backing interface{}
	switch dt {
	case tensor.Float64:
		b := make([]float64, rows*cols, rows*cols)
		b = b[:0]
		for i := 0; i < rows; i++ {
			for j := 0; j < len(M[i]); j++ {
				b = append(b, pixelWeight(M[i][j]))
			}
		}
		backing = b
	case tensor.Float32:
		b := make([]float32, rows*cols, rows*cols)
		b = b[:0]
		for i := 0; i < rows; i++ {
			for j := 0; j < len(M[i]); j++ {
				b = append(b, float32(pixelWeight(M[i][j])))
			}
		}
		backing = b
	}
	retVal = tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(backing))
	return
}

func prepareY(N []Label, dt tensor.Dtype) (retVal tensor.Tensor) {
	rows := len(N)
	cols := 10

	var backing interface{}
	switch dt {
	case tensor.Float64:
		b := make([]float64, rows*cols, rows*cols)
		b = b[:0]
		for i := 0; i < rows; i++ {
			for j := 0; j < 10; j++ {
				if j == int(N[i]) {
					b = append(b, 0.9)
				} else {
					b = append(b, 0.1)
				}
			}
		}
		backing = b
	case tensor.Float32:
		b := make([]float32, rows*cols, rows*cols)
		b = b[:0]
		for i := 0; i < rows; i++ {
			for j := 0; j < 10; j++ {
				if j == int(N[i]) {
					b = append(b, 0.9)
				} else {
					b = append(b, 0.1)
				}
			}
		}
		backing = b

	}
	retVal = tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(backing))
	return
}
func readLabelFile(r io.Reader, e error) (labels []Label, err error) {
	if e != nil {
		return nil, e
	}

	var (
		magic int32
		n     int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != labelMagic {
		return nil, os.ErrInvalid
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	labels = make([]Label, n)
	for i := 0; i < int(n); i++ {
		var l Label
		if err := binary.Read(r, binary.BigEndian, &l); err != nil {
			return nil, err
		}
		labels[i] = l
	}
	return labels, nil
}

func readImageFile(r io.Reader, e error) (imgs []RawImage, err error) {
	if e != nil {
		return nil, e
	}

	var (
		magic int32
		n     int32
		nrow  int32
		ncol  int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != imageMagic {
		return nil, err 
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &nrow); err != nil {
		return nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &ncol); err != nil {
		return nil, err
	}
	imgs = make([]RawImage, n)
	m := int(nrow * ncol)
	for i := 0; i < int(n); i++ {
		imgs[i] = make(RawImage, m)
		m_, err := io.ReadFull(r, imgs[i])
		if err != nil {
			return nil, err
		}
		if m_ != int(m) {
			return nil, os.ErrInvalid
		}
	}
	return imgs, nil
}
