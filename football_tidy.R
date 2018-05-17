library(tidyverse)
library(data.table)
library(caret)
library(yardstick)

data <- fread("~/Downloads/E0.csv")

data <- tbl_df(data)


data <- data %>% select(-Div, -FTHG, -FTAG) %>% select(1:20)



set.seed(101)
sample <- createDataPartition(data$FTR, p=0.80, list = FALSE)
train <- data[sample,]
test <- data[-sample,]


# Train a linear regression model
fit_rf <- train(FTR ~ ., method = "rf", data = train, trControl = trainControl(method = "boot"))

# Print the model object
fit_rf

# Train a linear regression model
fit_xg <- train(FTR ~ ., method = "xgbTree", data = train, trControl = trainControl(method = "boot"))

# Print the model object
fit_xg


results <- train %>% mutate(`Random forest` = predict(fit_rf, train))
results %>% select(FTR, `Random forest`)

metrics(results, truth = FTR, estimate = `Random forest`)
?metrics




