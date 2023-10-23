# Data Science: Regression Analysis
# Installation:
# Requirements:
- Anaconda
- Python 3.5
- Numpy
- Scipy
- Sci-kit learn
- Pandas
- matplotlib

# Models are trained based on two output parameters
- Case Fatality Ratio (CFR)
- CFR = Cumulative number of death / Cumulative number of cases

- COVID-19 Spreading Ratio (CSR)
- CSR = Cumulative number of cases / Total population

# Data collection strategy
- Data are collected from 57 locations from all over the world
- We have divided the locations into 5 categories where the locations in each category have the similar density of 65+ aged population.  
- Distribution of population density over 65+ aged people: 

# Input dataset:
- We have total 22 input features in the dataset.
- Input features are chosen based on environmental, biological, pollution, and demographical factors.
- Some input parameters influence CFR, some of them influence CSR and some have an impact on both CFR and CSR. 

# Input features dictate CFR:
- Population density of 65 or older people 
- Vitamin A deficiency 
- Vitamin D deficiency 
- Health care quality and access index
- Pollutant PM2.5 particle 
- Anemia 
- AB+ 
- AB-
- A+
- A- 
- B+ 
- B- 
- O+ 
- O- 

# Input features dictate CSR:
- Solar radiation
- Daily Temperature 
- Precipitation
- Population density
- Average Wind speed
- Relative humidity
- Stringency index
- UV index

# Input features dictate both CFR and CSR:
- Pollutant PM2.5 particle 

# Methodology:
- To train the input dataset, Artificial Neural Netwrok is used for regression.
- GridSearch technique is used to choose the hyperparameters, activation function, solver, and number and the size of hidden layers. 
- Four hidden layers are used to train the model with the sizes of 22, 20, 10, 5. 
- A regularization parameter alpha is used to prevent the trained model from overfitting. 
- Non-linear activation function (Relu) is used to fit the non-linear dataset.
- Solver Adam is used to handle the noisy dataset. 

# Results:
- For CFR:
- Training Error (MAE): 0.0350
- Testing Error (MAE): 0.0377

- For CSR:
- Training Error (MAE): 0.0184
- Testing Error (MAE): 0.0187
