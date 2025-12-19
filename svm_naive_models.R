library(caret)
library(smotefamily)
library(e1071)
library(tidyr)
library(dplyr)
library(ggplot2)

# ----------------- Load and Prepare Data -----------------
df <- read.csv("cleaned_stroke_data.csv")
df$stroke <- as.factor(df$stroke)

set.seed(123)
trainIndex <- sample(1:nrow(df), 0.8 * nrow(df))
trainData <- df[trainIndex, ]
testData  <- df[-trainIndex, ]

categorical_cols <- c("gender","hypertension","heart_disease",
                      "ever_married","work_type","Residence_type","smoking_status")

# Convert categorical to numeric for SMOTE
train_num <- trainData
for(col in categorical_cols){
  train_num[[col]] <- as.numeric(as.factor(train_num[[col]]))
}
train_num$stroke <- as.numeric(as.character(train_num$stroke))

# Apply SMOTE
set.seed(123)
smote_out <- SMOTE(
  X = train_num[, -which(names(train_num)=="stroke")],
  target = train_num$stroke,
  K = 5,
  dup_size = 5
)
train_smote <- smote_out$data
train_smote$stroke <- as.factor(train_smote$class)
train_smote$class <- NULL

# Prepare test data
for(col in categorical_cols){
  testData[[col]] <- as.numeric(
    factor(testData[[col]], levels = unique(trainData[[col]]))
  )
}

# ----------------- SVM Model -----------------
svm_model <- svm(
  stroke ~ .,
  data = train_smote,
  kernel = "radial",
  class.weights = c("0" = 1, "1" = 3),
  probability = TRUE
)

svm_pred <- predict(svm_model, newdata = testData)

svm_cm <- confusionMatrix(svm_pred, testData$stroke, positive = "1")

cat("\n--- Support Vector Machine ---\n")
cat("Accuracy :", round(svm_cm$overall["Accuracy"], 4), "\n")
cat("Precision:", round(svm_cm$byClass["Precision"], 4), "\n")
cat("Recall   :", round(svm_cm$byClass["Recall"], 4), "\n")
cat("F1 Score :", round(svm_cm$byClass["F1"], 4), "\n")

# Confusion Matrix Plot (normalized)
svm_cm_df <- as.data.frame(svm_cm$table)
svm_cm_df$Prop <- svm_cm_df$Freq / rowSums(svm_cm$table)[svm_cm_df$Prediction]

ggplot(svm_cm_df, aes(x=Prediction, y=Reference, fill=Prop)) +
  geom_tile(color="white") +
  geom_text(aes(label=sprintf("%.2f", Prop)), size=6, fontface="bold") +
  scale_fill_gradient(low="#F5F5F5", high="#388E3C") +
  labs(
    title="SVM Confusion Matrix (Normalized)",
    x="Predicted Class",
    y="Actual Class"
  ) +
  theme_minimal()

# ----------------- Naive Bayes Model -----------------
nb_model <- naiveBayes(stroke ~ ., data = train_smote)
nb_pred <- predict(nb_model, newdata = testData)

nb_cm <- confusionMatrix(nb_pred, testData$stroke, positive = "1")

cat("\n--- Naive Bayes ---\n")
cat("Accuracy :", round(nb_cm$overall["Accuracy"], 4), "\n")
cat("Precision:", round(nb_cm$byClass["Precision"], 4), "\n")
cat("Recall   :", round(nb_cm$byClass["Recall"], 4), "\n")
cat("F1 Score :", round(nb_cm$byClass["F1"], 4), "\n")

# Confusion Matrix Plot (normalized)
nb_cm_df <- as.data.frame(nb_cm$table)
nb_cm_df$Prop <- nb_cm_df$Freq / rowSums(nb_cm$table)[nb_cm_df$Prediction]

ggplot(nb_cm_df, aes(x=Prediction, y=Reference, fill=Prop)) +
  geom_tile(color="white") +
  geom_text(aes(label=sprintf("%.2f", Prop)), size=6, fontface="bold") +
  scale_fill_gradient(low="#F5F5F5", high="#FF8A65") +
  labs(
    title="Naive Bayes Confusion Matrix (Normalized)",
    x="Predicted Class",
    y="Actual Class"
  ) +
  theme_minimal()

# ----------------- PCA Visualization for SVM -----------------
X <- train_smote[, setdiff(names(train_smote), "stroke")]
y <- train_smote$stroke
pca <- prcomp(X, scale. = TRUE)

pca_df <- data.frame(
  PC1 = pca$x[, 1],
  PC2 = pca$x[, 2],
  stroke = y
)

grid <- expand.grid(
  PC1 = seq(min(pca_df$PC1), max(pca_df$PC1), length.out = 200),
  PC2 = seq(min(pca_df$PC2), max(pca_df$PC2), length.out = 200)
)

svm_pca <- svm(
  stroke ~ PC1 + PC2,
  data = pca_df,
  kernel = "radial",
  cost = 1,
  gamma = 0.5
)

grid$svm_pred <- predict(svm_pca, grid)

ggplot() +
  geom_tile(data = grid, aes(x = PC1, y = PC2, fill = svm_pred), alpha = 0.25) +
  geom_point(data = pca_df, aes(x = PC1, y = PC2, color = stroke), size = 1.5) +
  labs(
    title = "2D PCA Projection with SVM Decision Boundary",
    subtitle = "First two principal components",
    x = "PC1",
    y = "PC2"
  ) +
  scale_fill_manual(values = c("0" = "#BBDEFB", "1" = "#FFCDD2")) +
  scale_color_manual(values = c("0" = "blue", "1" = "red")) +
  theme_minimal(base_size = 13) +
  guides(fill = "none")

# ----------------- Feature Densities for Naive Bayes -----------------
plot_df <- train_smote %>%
  select(stroke, age, avg_glucose_level, bmi) %>%
  pivot_longer(
    cols = c(age, avg_glucose_level, bmi),
    names_to = "feature",
    values_to = "value"
  )

ggplot(plot_df, aes(x = value, fill = stroke)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~ feature, scales = "free") +
  labs(
    title = "Class-Conditional Feature Distributions (Naive Bayes)",
    subtitle = "Likelihoods P(feature | stroke)",
    x = "Feature value",
    y = "Density",
    fill = "Stroke"
  ) +
  scale_fill_manual(values = c("0" = "#90CAF9", "1" = "#EF9A9A")) +
  theme_minimal(base_size = 13)
