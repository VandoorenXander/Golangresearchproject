package main

import (
	"fmt"
	"log"
	"os"

	"github.com/tobgu/qframe"
	
)

func main() {
	csvfile, err := os.Open("./data/fashion-mnist.csv")
	if err != nil {
		log.Fatal(err)
	}
	df := qframe.ReadCSV(csvfile)
	dffilter := df.Filter(qframe.And(
		qframe.Filter{Column:"label",Comparator: "=",Arg:2},
		qframe.Filter{Column:"pixel2",Comparator: "=",Arg:0},
	))
	fmt.Println(dffilter.Select("label"))
	xdf:=df.Select("label")
	fmt.Println(xdf)
	// ydf := df.Drop("label")
	// fmt.Println(ydf)
	
	// fmt.Println(df.Select("label"))

}
