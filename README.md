# Time series forecast for net sales (by route / global)

This notebook shows the entire flow as a concept test to apply a forecast for net sales in a veterinary company (VC).
This VC has an ETL system already, that means that the retreived info is already cleaned.

This project is not focused on show how EDA process works for this case,
is to show how to implement a workflow by applying LSTM model instead.

---
## Stack
> - Python 3.11
> - Tensorflow
> - Go
> - Pandas
> - Seaborn / Matplotlib
> - Numpy
> - and adds...

All LSTm model creating, training and testing is gonna be running and developing
in python, while implementation is gonne be working on Golang.
This is mainly due to the speed that Golang has natively.

Althoguh this is a proof of concept (because the original database is running on postgres)
the entire workflow is the same. in order to run it on production environment, its necesary
to connect a database by modifying config file ```config.py```
---
## Project Structure
```
├── .dockerignore
├── .gitignore
├── README.md
│
├── ml/ #researching environment
│   ├── data/ #local datasets csv
│   ├── Dockerfile
│   ├── main.py #orchestrator
│   └── requirements.txt
│
├── artifacts/ #bridge between go and python
│   ├── lstm_sales.onnx #graph and model weights
│   └── scaler_params.json #universal scaler
│
└── service-go/ #inference service for production
    ├── cmd/
    │   └── api/
    │       └── main.go #http endpoint service
    ├── internal/
    │   ├── inference/ #load and exec with onnx
    │   └── preprocessor/ #scale new data by using universal scaler
    ├── Dockerfile
    ├── go.mod
    └── go.sum
```
## How to run
