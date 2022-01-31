package main

import (
	"fmt"
	"log"
	"os"
	// "encoding/csv"
	// "github.com/go-gota/gota"
	"github.com/go-gota/gota/dataframe"
	// "github.com/go-gota/gota/series"
)
 func main(){
	csvfile, err := os.Open("./data/fashion-mnist.csv")
	if err != nil {
		log.Fatal(err)
	}
	df := dataframe.ReadCSV(csvfile)
	xdf := df.Select("label")
	fmt.Println(xdf)
	ydf:=df.Drop("label")
	fmt.Println(ydf)
	xdffilter := xdf.Filter(dataframe.F{Colname:"label",Comparator: "==",Comparando: "9"})
	fmt.Println(xdffilter)

	// fmt.Println(df)
 }