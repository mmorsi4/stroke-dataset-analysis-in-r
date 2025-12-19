###############################################################################
#                          1st Model Decision Tree 
###############################################################################

library(rpart)
library(rpart.plot)
library(smotefamily)
library(caret)

df <- read.csv("cleaned_stroke_data.csv")
df$stroke <- as.factor(df$stroke)

# Load selected features
selected_features <- readRDS("selected_features.rds")
selected_features <- c(selected_features, "stroke")
df <- df[, selected_features]

# Split 80/20
set.seed(123)
trainIndex <- sample(1:nrow(df), 0.8 * nrow(df))
trainData <- df[trainIndex, ]
testData  <- df[-trainIndex, ]

# Identify categorical cols
categorical_cols <- intersect(
  c("gender","hypertension","heart_disease","ever_married",
    "work_type","Residence_type","smoking_status"),
  selected_features
)

# Convert categorical → numeric for SMOTE
train_num <- trainData
for(col in categorical_cols){
  train_num[[col]] <- as.numeric(as.factor(train_num[[col]]))
}

train_num$stroke <- as.numeric(as.character(train_num$stroke))

# SMOTE

set.seed(123)
smote_out <- SMOTE(
  X = train_num[, -which(names(train_num) == "stroke")],
  target = train_num$stroke,
  K = 5,
  dup_size = 3     
)

train_smote <- smote_out$data
train_smote$stroke <- as.factor(train_smote$class)
train_smote$class <- NULL

# Tuned Hyperparameters

loss_matrix <- matrix(c(0, 1, 10, 0), nrow=2, byrow=TRUE)
colnames(loss_matrix) <- rownames(loss_matrix) <- levels(train_smote$stroke)

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

rpart.plot(tree_model, type = 3, extra = 101, fallen.leaves = TRUE,
           main = "Decision Tree")

# Prepare test data
for(col in categorical_cols){
  testData[[col]] <- as.numeric(
    factor(testData[[col]], levels = unique(trainData[[col]]))
  )
}

# Predictions & Evaluation
pred <- predict(tree_model, newdata = testData, type = "class")
cm <- confusionMatrix(pred, testData$stroke, positive = "1")
print(cm)

cat("Accuracy :", round(cm$overall['Accuracy'], 4), "\n")
cat("Precision:", round(cm$byClass['Precision'], 4), "\n")
cat("Recall   :", round(cm$byClass['Recall'], 4), "\n")
cat("F1 Score :", round(cm$byClass['F1'], 4), "\n")
