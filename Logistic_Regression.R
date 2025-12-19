library(smotefamily)
library(caret)
library(pROC)
library(ggplot2)

# Load dataset
df <- read.csv("cleaned_stroke_data.csv")
df$stroke <- as.factor(df$stroke)

# Explicit selected features
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

# Train-test split (stratified)
set.seed(123)
trainIndex <- createDataPartition(df$stroke, p=0.8, list=FALSE)
trainData <- df[trainIndex, ]
testData  <- df[-trainIndex, ]

# Categorical columns
categorical_cols <- c(
  "hypertension",
  "heart_disease",
  "ever_married",
  "work_type",
  "smoking_status"
)

# Convert categorical → numeric for SMOTE
train_num <- trainData
for (col in categorical_cols) {
  train_num[[col]] <- as.numeric(as.factor(train_num[[col]]))
}
train_num$stroke <- as.numeric(as.character(train_num$stroke))

# Apply SMOTE
set.seed(123)
smote_out <- SMOTE(
  X = train_num[, -which(names(train_num) == "stroke")],
  target = train_num$stroke,
  K = 5,
  dup_size = 4
)

train_smote <- smote_out$data
train_smote$stroke <- as.factor(train_smote$class)
train_smote$class <- NULL

# Logistic Regression Model
log_model <- glm(stroke ~ ., data=train_smote, family=binomial)
summary(log_model)

# Prepare test data: match factor levels
for (col in categorical_cols) {
  testData[[col]] <- as.numeric(
    factor(testData[[col]], levels = levels(as.factor(trainData[[col]])))
  )
}

# Predict probabilities → classes
prob <- predict(log_model, newdata=testData, type="response")
pred_log <- ifelse(prob >= 0.5, "1", "0")
pred_log <- factor(pred_log, levels = c("0","1"))

# Make sure test labels are factors with same levels
testData$stroke <- factor(testData$stroke, levels = c("0","1"))

# Confusion Matrix (should now work)
cm_log <- confusionMatrix(pred_log, testData$stroke, positive = "1")
print(cm_log)

# Metrics
cat("Accuracy :", round(cm_log$overall['Accuracy'],4), "\n")
cat("Precision:", round(cm_log$byClass['Precision'],4), "\n")
cat("Recall   :", round(cm_log$byClass['Recall'],4), "\n")
cat("F1 Score :", round(cm_log$byClass['F1'],4), "\n")


# ROC Curve
roc_obj <- roc(testData$stroke, prob)
plot(roc_obj, main="ROC Curve – Logistic Regression", col="blue", lwd=3)
auc_val <- auc(roc_obj)
cat("AUC:", round(auc_val,4), "\n")

# Precision-Recall Barplot
ggplot(data.frame(prob=prob, stroke=testData$stroke), aes(x=prob, fill=stroke)) +
  geom_histogram(position="identity", bins=30, alpha=0.5) +
  ggtitle("Predicted Probabilities – Logistic Regression") +
  xlab("Predicted Probability") + ylab("Count") +
  theme_minimal()
