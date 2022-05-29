package main

import (
	"context"
	"fmt"
	"log"
	"os"

	// "strings"
	// "github.com/rocketlaunchr/dataframe-go"
	"github.com/rocketlaunchr/dataframe-go/imports"
)
var ctx = context.Background()
func main() {
	data, err := os.Open("./data/fashion-mnist.csv")
	if err != nil {
		log.Fatal(err)
	}
	df, err := imports.LoadFromCSV(ctx, data)
	// fmt.Print(df.Table())
	fmt.Println(df)	
}