###############################################################################
#                              Feature Selection
###############################################################################

# Load data after hypothesis testing
df <- read.csv("data_after_testing.csv")

# Feature evaluation table
feature_eval <- data.frame(
  feature = character(),
  p_value = numeric(),
  test_type = character(),
  decision = character(),
  reason = character(),
  stringsAsFactors = FALSE
)

# Fill with results from hypothesis testing:
feature_eval <- rbind(feature_eval,
                      data.frame(
                        feature = "age",
                        p_value = 2.2e-16,
                        test_type = "t-test",
                        decision = "KEEP",
                        reason = "Highly significant (p < 2.2e-16)"
                      ),
                      data.frame(
                        feature = "gender",
                        p_value = 0.52,
                        test_type = "t-test",
                        decision = "DROP",
                        reason = "Not significant (p = 0.52)"
                      ),
                      data.frame(
                        feature = "hypertension",
                        p_value = 3.6e-09,
                        test_type = "t-test & wilcoxon",
                        decision = "KEEP",
                        reason = "Highly significant, known medical risk factor"
                      ),
                      data.frame(
                        feature = "heart_disease",
                        p_value = 4.5e-08,
                        test_type = "t-test & wilcoxon",
                        decision = "KEEP",
                        reason = "Highly significant, known medical risk factor"
                      ),
                      data.frame(
                        feature = "ever_married",
                        p_value = 2.2e-16,
                        test_type = "t-test & wilcoxon",
                        decision = "KEEP",
                        reason = "Significant difference found"
                      ),
                      data.frame(
                        feature = "work_type",
                        p_value = 4.91e-10,
                        test_type = "ANOVA",
                        decision = "KEEP",
                        reason = "Highly significant overall differences (ANOVA p ≈ 4.91e-10)"
                      ),
                      data.frame(
                        feature = "Residence_type",
                        p_value = 0.27,
                        test_type = "t-test & wilcoxon",
                        decision = "DROP",
                        reason = "Not significant (p ≈ 0.27)"
                      ),
                      data.frame(
                        feature = "smoking_status",
                        p_value = 3.23e-06,
                        test_type = "ANOVA",
                        decision = "KEEP",
                        reason = "Significant differences (ANOVA p ≈ 3.23e-06)"
                      ),
                      data.frame(
                        feature = "avg_glucose_level",
                        p_value = 1.76e-08,  # t-test p-value
                        test_type = "t-test",
                        decision = "KEEP",
                        reason = "Highly significant (t-test p ≈ 1.76e-08)"
                      ),
                      data.frame(
                        feature = "bmi",
                        p_value = 0.000185,  # t-test p-value
                        test_type = "t-test",
                        decision = "KEEP",
                        reason = "Significant (t-test p ≈ 0.000185)"
                      )
)

# Selected features (excluding the dropped ones)
selected_features <- feature_eval$feature[feature_eval$decision == "KEEP"]

# Remove duplicates and decision variable
selected_features <- unique(selected_features)
selected_features <- selected_features[!selected_features %in% "stroke"]

# Save for model
saveRDS(selected_features, "selected_features.rds")

# Save the full evaluation table
write.csv(feature_eval, "feature_selection_evaluation.csv", row.names = FALSE)

cat("Feature Selection Complete!
Based on hypothesis testing results, selected", length(selected_features), "features:
", paste(selected_features, collapse = ", "))