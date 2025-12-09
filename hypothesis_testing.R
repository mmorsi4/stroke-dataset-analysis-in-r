###############################################################################
#                             Hypothesis Testing
###############################################################################

df <- read.csv("cleaned_stroke_data.csv", na.strings = c("N/A", "Unknown"))

# Prepare variables
df$stroke_group <- factor(df$stroke, levels = c(0,1), labels = c("no","yes"))
df$stroke_num   <- as.numeric(df$stroke)

df$gender         <- factor(df$gender)
df$hypertension   <- factor(df$hypertension)
df$heart_disease  <- factor(df$heart_disease)
df$ever_married   <- factor(df$ever_married)
df$work_type      <- factor(df$work_type)
df$Residence_type <- factor(df$Residence_type)
df$smoking_status <- factor(df$smoking_status)

# Helper functions
safe_t_test <- function(formula, data) {
  print(t.test(formula, data = data))
}
safe_wilcox_test <- function(formula, data) {
  print(wilcox.test(formula, data = data))
}

# Age: significant difference between stroke vs non-stroke groups
# p < 2.2e-16 (both t-test and Wilcoxon)
safe_t_test(age ~ stroke_group, data = df)
safe_wilcox_test(age ~ stroke_group, data = df)

# Average glucose level: significant difference
# t-test p ≈ 1.76e-08, Wilcoxon p ≈ 4.6e-07
safe_t_test(avg_glucose_level ~ stroke_group, data = df)
safe_wilcox_test(avg_glucose_level ~ stroke_group, data = df)

# BMI: small but significant difference
# t-test p ≈ 0.000185, Wilcoxon p ≈ 7e-05
safe_t_test(bmi ~ stroke_group, data = df)
safe_wilcox_test(bmi ~ stroke_group, data = df)

# Gender: NOT significant
# t-test p ≈ 0.52, Wilcoxon p ≈ 0.52
safe_t_test(stroke_num ~ gender, data = df)
safe_wilcox_test(stroke_num ~ gender, data = df)

# Hypertension: highly significant
safe_t_test(stroke_num ~ hypertension, data = df)
safe_wilcox_test(stroke_num ~ hypertension, data = df)

# Heart disease: highly significant
safe_t_test(stroke_num ~ heart_disease, data = df)
safe_wilcox_test(stroke_num ~ heart_disease, data = df)

# Ever married: significant difference
safe_t_test(stroke_num ~ ever_married, data = df)
safe_wilcox_test(stroke_num ~ ever_married, data = df)

# Residence type: NOT significant
# t-test p ≈ 0.27, Wilcoxon p ≈ 0.27
safe_t_test(stroke_num ~ Residence_type, data = df)
safe_wilcox_test(stroke_num ~ Residence_type, data = df)

# Work type: significant overall differences (ANOVA p ≈ 4.91e-10)
aov_work <- aov(stroke_num ~ work_type, data = df)
summary(aov_work)
TukeyHSD(aov_work)

# Smoking status: significant differences (ANOVA p ≈ 3.23e-06)
aov_smoke <- aov(stroke_num ~ smoking_status, data = df)
summary(aov_smoke)
TukeyHSD(aov_smoke)

# Non-significant predictors:
#   - gender (p > 0.05)
#   - Residence_type (p > 0.05)
#
# Decision: DROP both variables.

df$gender <- NULL
df$Residence_type <- NULL

write.csv(df, "data_after_testing.csv", row.names = FALSE)