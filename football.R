library(data.table)
library(dplyr)
library(h2o) # for H2O Machine Learning
library(lime) # for Machine Learning Interpretation
library(mlbench) # for Datasets

data <- fread("~/Downloads/E0.csv")
# data <- h2o.importFile(path = normalizePath("~/Downloads/E0.csv"))
data %>% as_tibble()

data <- data %>%  select(-Div, -Date, -FTHG, -FTAG)
# , -FTHG, -FTAG, -HTHG, -HTAG, -HTR) 
#, -HomeTeam, -AwayTeam, -HTR, -Referee)
data$HomeTeam <- as.factor(data$HomeTeam)
data$AwayTeam <- as.factor(data$AwayTeam)
data$Referee <- as.factor(data$Referee)
data$FTR <- as.factor(data$FTR)

#data <- data %>% select(1:16) %>% as_tibble()

# Your lucky seed here ...
n_seed = 12345

dim(data)
data %>% as_tibble()


# Create target and feature list
target = "FTR" # Result
features = setdiff(colnames(data), target)
print(features)


# Start a local H2O cluster (JVM)
h2o.init()

# H2O dataframe
h_data <-  as.h2o(data)


# Split Train/Test
h_split = h2o.splitFrame(h_data, ratios = 0.75, seed = n_seed)
h_train = h_split[[1]] # 75% for modelling
h_test = h_split[[2]] # 25% for evaluation




# Train a Default H2O GBM model
model_gbm = h2o.gbm(x = features,
                    y = target,
                    training_frame = h_train,
                    model_id = "my_gbm",
                    seed = n_seed)
print(model_gbm)


# Evaluate performance on test
h2o.performance(model_gbm, newdata = h_test)


# Train multiple H2O models with H2O AutoML
# Stacked Ensembles will be created from those H2O models
# You tell H2O ...
#     1) how much time you have and/or 
#     2) how many models do you want
# Note: H2O deep learning algo on multi-core is stochastic
model_automl = h2o.automl(x = features,
                          y = target,
                          training_frame = h_train,
                          nfolds = 5,               # Cross-Validation
                          max_runtime_secs = 120,   # Max time
                          max_models = 100,         # Max no. of models
                          stopping_metric = "misclassification", # Metric to optimize
                          project_name = "my_automl",
                          exclude_algos = NULL,     # If you want to exclude any algo 
                          seed = n_seed)


model_automl@leaderboard # It seems Deep learning is the best

model_automl@leader

