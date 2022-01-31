package main

import (
	"context"
	"fmt"
	"os"
	"log"
	"strings"
	"github.com/rocketlaunchr/dataframe-go"
	"github.com/rocketlaunchr/dataframe-go/imports"
)
var ctx = context.Background()
func main() {
	data, err := os.Open("./data/fashion-mnist_test.csv")
	if err != nil {
		log.Fatal(err)
	}
	df, err := imports.LoadFromCSV(ctx, strings.NewReader(data))
	fmt.Print(df.Table())
}