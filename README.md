# MLT

###preprocess.py
- Input: message and orderbook files
- Output: pandas dataframe in csv format after some preliminary data transformation

###gen_features.py
- Input: output from preprocess.py
- Output: training data with features

###train.py
- Input: features
- Output: model performances

python preprocess.py data/AMZN/message.csv data/AMZN/orderbook.csv output.csv
python gen_features.py df.csv features.csv
python train.py features.csv
