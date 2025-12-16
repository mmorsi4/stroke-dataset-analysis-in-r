###############################################################################
#        MACHINE LEARNING: SVM & NAIVE BAYES (AFTER HYPOTHESIS TESTING)
###############################################################################

# Load required libraries
library(caret)
library(smotefamily)
library(e1071)

###############################################################################
# Load data AFTER hypothesis testing
###############################################################################

df <- read.csv("data_after_testing.csv")
df$stroke <- as.factor(df$stroke)

###############################################################################
# Train-test split (80% / 20%)
###############################################################################

set.seed(123)
trainIndex <- sample(seq_len(nrow(df)), size = 0.8 * nrow(df))
trainData <- df[trainIndex, ]
testData  <- df[-trainIndex, ]

###############################################################################
# Identify categorical columns (UPDATED: dropped non-significant ones)
###############################################################################

categorical_cols <- c(
  "hypertension",
  "heart_disease",
  "ever_married",
  "work_type",
  "smoking_status"
)

###############################################################################
# Apply SMOTE (CORRECTED)
###############################################################################

# Start from training data
train_num <- trainData

# REMOVE non-predictor columns created during hypothesis testing
train_num$stroke_group <- NULL
train_num$stroke_num   <- NULL

# Convert ALL predictors to numeric
for(col in names(train_num)){
  if(col != "stroke"){
    train_num[[col]] <- as.numeric(as.factor(train_num[[col]]))
  }
}

# Convert target to numeric for SMOTE
train_num$stroke <- as.numeric(as.character(train_num$stroke))

set.seed(123)
smote_out <- SMOTE(
  X = train_num[, -which(names(train_num) == "stroke")],
  target = train_num$stroke,
  K = 5,
  dup_size = 5
)

###############################################################################
# Rebuild SMOTE training dataset
###############################################################################

train_smote <- smote_out$data
train_smote$stroke <- as.factor(train_smote$class)
train_smote$class <- NULL

###############################################################################
# Prepare test data (same encoding as training)
###############################################################################

for(col in categorical_cols){
  testData[[col]] <- as.numeric(
    factor(testData[[col]], levels = unique(trainData[[col]]))
  )
}

###############################################################################
# MODEL 1: SUPPORT VECTOR MACHINE (Cost-Sensitive)
###############################################################################

svm_model <- svm(
  stroke ~ .,
  data = train_smote,
  kernel = "radial",
  class.weights = c("0" = 1, "1" = 3),
  probability = TRUE
)

svm_pred <- predict(svm_model, newdata = testData)

svm_cm <- confusionMatrix(svm_pred, testData$stroke, positive = "1")

cat("\n================ SVM RESULTS ================\n")
cat("Accuracy :", round(svm_cm$overall["Accuracy"], 4), "\n")
cat("Precision:", round(svm_cm$byClass["Precision"], 4), "\n")
cat("Recall   :", round(svm_cm$byClass["Recall"], 4), "\n")
cat("F1 Score :", round(svm_cm$byClass["F1"], 4), "\n")

###############################################################################
# MODEL 2: NAIVE BAYES
###############################################################################

nb_model <- naiveBayes(
  stroke ~ .,
  data = train_smote
)

nb_pred <- predict(nb_model, newdata = testData)

nb_cm <- confusionMatrix(nb_pred, testData$stroke, positive = "1")

cat("\n============= NAIVE BAYES RESULTS =============\n")
cat("Accuracy :", round(nb_cm$overall["Accuracy"], 4), "\n")
cat("Precision:", round(nb_cm$byClass["Precision"], 4), "\n")
cat("Recall   :", round(nb_cm$byClass["Recall"], 4), "\n")
cat("F1 Score :", round(nb_cm$byClass["F1"], 4), "\n")

###############################################################################
# END OF SCRIPT
###############################################################################
