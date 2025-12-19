###############################################################################
#                          1st Model – Decision Tree 
###############################################################################

library(rpart)
library(rpart.plot)
library(smotefamily)
library(caret)
library(ggplot2)

# Load dataset
df <- read.csv("cleaned_stroke_data.csv")
df$stroke <- as.factor(df$stroke)

# Explicit feature list (prevents mismatch bugs)
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

# Stratified 80/20 split
set.seed(123)
trainIndex <- createDataPartition(df$stroke, p = 0.8, list = FALSE)
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

# Prepare numeric data for SMOTE
train_num <- trainData
for (col in categorical_cols) {
  train_num[[col]] <- as.numeric(as.factor(train_num[[col]]))
}

# Target to numeric for SMOTE
train_num$stroke <- as.numeric(as.character(train_num$stroke))

# Apply SMOTE
set.seed(123)
smote_out <- SMOTE(
  X = train_num[, -which(names(train_num) == "stroke")],
  target = train_num$stroke,
  K = 5,
  dup_size = 3
)

# Rebuild SMOTE-balanced dataset
train_smote <- smote_out$data
train_smote$stroke <- as.factor(train_smote$class)
train_smote$class <- NULL

# Cost-sensitive loss matrix (penalize FN heavily)
loss_matrix <- matrix(
  c(0, 1,
    10, 0),
  nrow = 2,
  byrow = TRUE
)
colnames(loss_matrix) <- rownames(loss_matrix) <- levels(train_smote$stroke)

# Train Decision Tree
tree_model <- rpart(
  stroke ~ .,
  data = train_smote,
  method = "class",
  parms = list(loss = loss_matrix),
  control = rpart.control(
    cp = 0.001,
    minsplit = 50,
    minbucket = 10,
    maxdepth = 10
  )
)

# Plot tree
rpart.plot(
  tree_model,
  type = 3,
  extra = 101,
  fallen.leaves = TRUE,
  main = "Decision Tree for Stroke Prediction"
)

# Encode test data using training levels (CRITICAL FIX)
for (col in categorical_cols) {
  testData[[col]] <- as.numeric(
    factor(testData[[col]],
           levels = levels(as.factor(trainData[[col]])))
  )
}

# Predictions
pred_dt <- predict(tree_model, newdata = testData, type = "class")

# Evaluation
cm_dt <- confusionMatrix(pred_dt, testData$stroke, positive = "1")
print(cm_dt)

cm_df <- as.data.frame(cm_dt$table)

ggplot(cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6, fontface = "bold") +
  scale_fill_gradient(low = "#F5F5F5", high = "#1976D2") +
  labs(
    title = "Confusion Matrix – Decision Tree",
    subtitle = "Cost-Sensitive Learning with SMOTE",
    x = "Predicted Class",
    y = "Actual Class"
  ) +
  theme_minimal()

# Metrics
cat("Accuracy :", round(cm_dt$overall["Accuracy"], 4), "\n")
cat("Precision:", round(cm_dt$byClass["Precision"], 4), "\n")
cat("Recall   :", round(cm_dt$byClass["Recall"], 4), "\n")
cat("F1 Score :", round(cm_dt$byClass["F1"], 4), "\n")
