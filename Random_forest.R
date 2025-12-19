###############################################################################
#                         2nd Model Random Forest 
###############################################################################
# Load necessary libraries
library(randomForest)
library(smotefamily)
library(caret)

# Load the dataset and ensure target is a factor
df <- read.csv("cleaned_stroke_data.csv")
df$stroke <- as.factor(df$stroke)

# Load selected features and subset the dataset
selected_features <- readRDS("selected_features.rds")
selected_features <- c(selected_features, "stroke")
df <- df[, selected_features]

# Split data into training (80%) and testing (20%)
set.seed(123)
trainIndex <- sample(1:nrow(df), 0.8 * nrow(df))
trainData  <- df[trainIndex, ]
testData   <- df[-trainIndex, ]

# Identify categorical columns to convert for SMOTE
categorical_cols <- intersect(
  c("gender","hypertension","heart_disease","ever_married",
    "work_type","Residence_type","smoking_status"),
  selected_features
)

train_num <- trainData

# Convert categorical features to numeric
for(col in categorical_cols){
  train_num[[col]] <- as.numeric(as.factor(train_num[[col]]))
}

# Convert target to numeric for SMOTE
train_num$stroke <- as.numeric(as.character(train_num$stroke))

# Apply SMOTE to balance classes
set.seed(123)
smote_out <- SMOTE(
  X = train_num[, -which(names(train_num) == "stroke")],
  target = train_num$stroke,
  K = 5,
  dup_size = 3
)

# Extract SMOTE-balanced training dataset
train_smote <- smote_out$data
train_smote$stroke <- as.factor(train_smote$class)
train_smote$class <- NULL

# Train Random Forest model with class weights for imbalanced data
set.seed(123)
rf_model <- randomForest(
  stroke ~ .,
  data = train_smote,
  ntree = 300,                                
  mtry = floor(sqrt(ncol(train_smote)-1)),   
  importance = TRUE,
  classwt = c("0" = 1, "1" = 8)             
)

print(rf_model)

# Prepare test data using same encoding as training
for(col in categorical_cols){
  testData[[col]] <- as.numeric(
    factor(testData[[col]], levels = unique(trainData[[col]]))
  )
}

# Predict probabilities for test set
probs <- predict(rf_model, newdata = testData, type = "prob")[,2]

# Use threshold 0.3 to increase sensitivity
pred_rf <- ifelse(probs > 0.3, "1", "0")
pred_rf <- factor(pred_rf, levels = c("0","1"))

# Evaluate the model
cm_rf <- confusionMatrix(pred_rf, testData$stroke, positive = "1")
print(cm_rf)

# Print key metrics
cat("Accuracy :", round(cm_rf$overall['Accuracy'], 4), "\n")
cat("Precision:", round(cm_rf$byClass['Precision'], 4), "\n")
cat("Recall   :", round(cm_rf$byClass['Recall'], 4), "\n")
cat("F1 Score :", round(cm_rf$byClass['F1'], 4), "\n")

# Plot feature importance
varImpPlot(rf_model, main="Random Forest Feature Importance")

# ------------------ Analysis / Conclusion ------------------
cat("\nAnalysis:\n")
cat("The Random Forest model achieves an overall accuracy of approximately", 
    round(cm_rf$overall['Accuracy'], 2), "on the test set.\n")
cat("Recall for the stroke-positive class is", round(cm_rf$byClass['Recall'], 2),
    "indicating that the model correctly identifies about", 
    round(cm_rf$byClass['Recall']*100, 1), "% of actual stroke cases.\n")
cat("Precision is lower, showing that many predicted stroke cases are false positives.\n")
cat("Using SMOTE, class weights, and a lower threshold improved recall, which is crucial\nfor medical diagnosis tasks where missing a stroke case is costly.\n")
cat("Feature importance indicates which variables are most influential in predicting stroke,\nallowing further insights for medical analysis.\n")
