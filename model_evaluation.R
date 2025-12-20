###############################################################################
#                 Evaluation – ROC & AUC Comparison
###############################################################################

library(pROC)
library(caret)

# ----------------- Load trained models -----------------
source("Descision_Tree.R")
source("Logistic_Regression.R")
source("Random_forest.R")
source("svm_naive_models.R")  # contains both SVM and Naive Bayes

# ----------------- Load and prepare data -----------------
df <- read.csv("cleaned_stroke_data.csv")
df$stroke <- as.factor(df$stroke)

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

# Same train-test split for all models
set.seed(123)
trainIndex <- createDataPartition(df$stroke, p = 0.8, list = FALSE)
trainData <- df[trainIndex, ]
testData  <- df[-trainIndex, ]

categorical_cols <- c(
  "hypertension",
  "heart_disease",
  "ever_married",
  "work_type",
  "smoking_status"
)

# Encode categorical columns in test set using training levels
for (col in categorical_cols) {
  testData[[col]] <- as.numeric(
    factor(testData[[col]], levels = levels(as.factor(trainData[[col]])))
  )
}

# Ensure target factor levels match
testData$stroke <- factor(testData$stroke, levels = c("0", "1"))

# ----------------- Predicted probabilities -----------------
prob_log <- predict(log_model, testData, type = "response")
prob_dt  <- predict(tree_model, testData, type = "prob")[,2]
prob_rf  <- predict(rf_model, testData, type = "prob")[,2]
prob_nb  <- predict(nb_model, testData, type = "raw")[,2]
prob_svm <- attr(predict(svm_model, testData, probability = TRUE), "probabilities")[,2]

# ----------------- ROC Curves -----------------
roc_log <- roc(testData$stroke, prob_log)
roc_dt  <- roc(testData$stroke, prob_dt)
roc_rf  <- roc(testData$stroke, prob_rf)
roc_nb  <- roc(testData$stroke, prob_nb)
roc_svm <- roc(testData$stroke, prob_svm)

# ----------------- Open new device to show plot -----------------
dev.new()  # 🔑 this makes sure the plot window opens

plot(roc_log, col="blue", lwd=2, main="ROC Curve Comparison")
plot(roc_dt, col="red", lwd=2, add=TRUE)
plot(roc_rf, col="green", lwd=2, add=TRUE)
plot(roc_nb, col="purple", lwd=2, add=TRUE)
plot(roc_svm, col="orange", lwd=2, add=TRUE)

legend(
  "bottomright",
  legend = c(
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Naive Bayes",
    "SVM"
  ),
  col = c("blue","red","green","purple","orange"),
  lwd = 2
)

# ----------------- AUC Table -----------------
auc_results <- data.frame(
  Model = c(
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Naive Bayes",
    "SVM"
  ),
  AUC = c(
    auc(roc_log),
    auc(roc_dt),
    auc(roc_rf),
    auc(roc_nb),
    auc(roc_svm)
  )
)

print(auc_results)
