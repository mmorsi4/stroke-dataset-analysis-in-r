###############################################################################
#                         2nd Model Random Forest 
###############################################################################

library(randomForest)
library(smotefamily)
library(caret)

# Load dataset
df <- read.csv("cleaned_stroke_data.csv")
df$stroke <- as.factor(df$stroke)

# Explicitly define selected features (avoid mismatch)
selected_features <- c(
  "age",
  "hypertension",
  "heart_disease",
  "ever_married",
  "work_type",
  "smoking_status",
  "avg_glucose_level",
  "bmi",
  "stroke"
)

df <- df[, selected_features]

# Train-test split (80/20)
set.seed(123)
trainIndex <- createDataPartition(df$stroke, p = 0.8, list = FALSE)
trainData <- df[trainIndex, ]
testData  <- df[-trainIndex, ]

# Categorical columns (ONLY from selected features)
categorical_cols <- c(
  "hypertension",
  "heart_disease",
  "ever_married",
  "work_type",
  "smoking_status"
)

# Copy training data for SMOTE
train_num <- trainData

# Encode categorical variables numerically for SMOTE
for (col in categorical_cols) {
  train_num[[col]] <- as.numeric(as.factor(train_num[[col]]))
}

# Convert target to numeric (required by SMOTE)
train_num$stroke <- as.numeric(as.character(train_num$stroke))

# Apply SMOTE
set.seed(123)
smote_out <- SMOTE(
  X = train_num[, -which(names(train_num) == "stroke")],
  target = train_num$stroke,
  K = 5,
  dup_size = 3
)

# Reconstruct SMOTE dataset
train_smote <- smote_out$data
train_smote$stroke <- as.factor(train_smote$class)
train_smote$class <- NULL

# Train Random Forest
set.seed(123)
rf_model <- randomForest(
  stroke ~ .,
  data = train_smote,
  ntree = 300,
  mtry = floor(sqrt(ncol(train_smote) - 1)),
  importance = TRUE,
  classwt = c("0" = 1, "1" = 8)
)

print(rf_model)

# Encode test set using training levels (CRITICAL FIX)
for (col in categorical_cols) {
  testData[[col]] <- as.numeric(
    factor(testData[[col]],
           levels = levels(as.factor(trainData[[col]])))
  )
}

# Predict probabilities
probs <- predict(rf_model, newdata = testData, type = "prob")[, 2]

# Lower threshold to boost recall
pred_rf <- ifelse(probs > 0.3, "1", "0")
pred_rf <- factor(pred_rf, levels = c("0", "1"))

# Evaluation
cm_rf <- confusionMatrix(pred_rf, testData$stroke, positive = "1")
print(cm_rf)

cm_df <- as.data.frame(cm_rf$table)

ggplot(cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6, fontface = "bold") +
  scale_fill_gradient(low = "#F5F5F5", high = "#D32F2F") +
  labs(
    title = "Confusion Matrix – Random Forest",
    subtitle = "Threshold = 0.3 (Recall-Optimized)",
    x = "Predicted Label",
    y = "Actual Label"
  ) +
  theme_minimal()

# Metrics
cat("Accuracy :", round(cm_rf$overall["Accuracy"], 4), "\n")
cat("Precision:", round(cm_rf$byClass["Precision"], 4), "\n")
cat("Recall   :", round(cm_rf$byClass["Recall"], 4), "\n")
cat("F1 Score :", round(cm_rf$byClass["F1"], 4), "\n")

# Feature importance
varImpPlot(rf_model, main = "Random Forest Feature Importance")

###############################################################################
# Analysis / Conclusion
###############################################################################
cat("\nAnalysis:\n")
cat("The Random Forest model demonstrates strong recall performance after\n")
cat("feature selection, SMOTE balancing, class weighting, and threshold tuning.\n")
cat("This configuration prioritizes sensitivity, which is appropriate for\n")
cat("stroke prediction where false negatives are costly.\n")
