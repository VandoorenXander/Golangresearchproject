#installing golang
FROM golang:1.17.6-alpine 
#workdir directory
WORKDIR /go/src/researchproject/

COPY . .

#installing the packages needed for the project
RUN go get github.com/pkg/errors
RUN go get gopkg.in/cheggaaa/pb.v1
RUN go get gorgonia.org/gorgonia
RUN go get gorgonia.org/tensor
RUN go get github.com/rocketlaunchr/dataframe-go
RUN go get github.com/rocketlaunchr/dataframe-go/imports
RUN go get github.com/tobgu/qframe
RUN go get github.com/go-gota/gota/dataframe
RUN go get github.com/go-gota/gota/series
RUN go get gonum.org/v1/gonum




