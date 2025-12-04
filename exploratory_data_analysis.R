###############################################################################
#                         Exploratory Data Analysis (EDA)
###############################################################################

df <- read.csv("cleaned_stroke_data.csv", na.strings = c("N/A", "Unknown"))

numeric_cols <- c("age", "avg_glucose_level", "bmi", "stroke")
df_numeric <- df[, numeric_cols]

# Create directory if it does not exist
if(!dir.exists("plots")) dir.create("plots")

# Pastel colors for all barplots
pastel_colors <- c("#FFB3BA", "#BAE1FF", "#BAFFC9", "#FFFFBA", "#FFDFBA")

png("plots/pairwise_numeric_vars.png", width=1000, height=1000)
# Pairwise plot for numeric variables and target variable
pairs(df_numeric, 
      main = "Pairwise Plots of Numeric Variables",
      pch = 21,
      bg = c("red", "blue")[as.factor(df$stroke)]) # color by stroke

# Conclusion: Stroke is proportional to age
# Conclusion: High BMI is tied to higher ages (Higher BMI -> Most likely high age)
dev.off()

png("plots/boxplots_numeric_vs_stroke.png", width=1000, height=1000)
par(mfrow=c(1,3))

boxplot(age ~ stroke, data=df, main="Age vs Stroke")
# Conclusion: Stroke is CONFIRMED to be proportional to age

boxplot(avg_glucose_level ~ stroke, data=df, main="Avg Glucose vs Stroke")
# Conclusion: The average glucose levels are generally higher for patients with a stroke

boxplot(bmi ~ stroke, data=df, main="BMI vs Stroke")
# Conclusion: Stroke seems to increase for intermediate values of BMI (not too high or too low)
dev.off()

png("plots/barplots_gender_hypertension_heartdisease.png", width=1000, height=1000)
par(mfrow=c(1,3))

table(df$gender, df$stroke)
barplot(table(df$gender, df$stroke), beside=TRUE, legend=TRUE, main="Gender vs Stroke", xlab="Stroke", ylab="Count", col=pastel_colors[1:2])
# Conclusion: Females may seem to have more strokes but in reality the hypothesis testing shows that there is no correleation
# This may happen due to data imbalance (more females than males)

table(df$hypertension, df$stroke)
barplot(table(df$hypertension, df$stroke), beside=TRUE, legend=TRUE, 
        main="Hypertension vs Stroke", xlab="Stroke", ylab="Count", col=pastel_colors[3:4])
# Conclusion: Patients with hypertension are more likely to have a stroke compared to patients with no hypertension

table(df$heart_disease, df$stroke)
barplot(table(df$heart_disease, df$stroke), beside=TRUE, legend=TRUE, 
        main="Heart Disease vs Stroke", xlab="Stroke", ylab="Count", col=pastel_colors[5:1])
# Conclusion: Patients with heart disease are more likely to have a stroke compared to patients with no heart disease
dev.off()

png("plots/barplots_evermarried_worktype_residencetype_smoking.png", width=1000, height=1000)
par(mfrow=c(2, 2))

# Ever Married vs Stroke
table(df$ever_married, df$stroke)
barplot(table(df$ever_married, df$stroke), beside=TRUE, legend=TRUE, 
        main="Ever Married vs Stroke", xlab="Stroke", ylab="Count", col=pastel_colors[1:2])
# Conclusion: No apparent correlation

# Work Type vs Stroke
table(df$work_type, df$stroke)
barplot(table(df$work_type, df$stroke), beside=TRUE, legend=TRUE, 
        main="Work Type vs Stroke", xlab="Stroke", ylab="Count", col=pastel_colors[1:5])
# Conclusion: No apparent correlation

# Residence Type vs Stroke
table(df$Residence_type, df$stroke)
barplot(table(df$Residence_type, df$stroke), beside=TRUE, legend=TRUE, 
        main="Residence Type vs Stroke", xlab="Stroke", ylab="Count", col=pastel_colors[2:3])
# Conclusion: No correlation (backed by hypothesis testing)

# Smoking Status vs Stroke
table(df$smoking_status, df$stroke)
barplot(table(df$smoking_status, df$stroke), beside=TRUE, legend=TRUE, 
        main="Smoking Status vs Stroke", xlab="Stroke", ylab="Count", col=pastel_colors[1:5])
# Conclusion: No apparent correlation
dev.off()

png("plots/histograms_age_glucose_bmi.png", width=1000, height=1000)
par(mfrow=c(3, 1))

# Age Histogram
hist(df$age, breaks=20, col="skyblue", main="Age Distribution with Density", xlab="Age", freq=FALSE)
lines(density(df$age, na.rm=TRUE), col="red", lwd=2)
# Age seems to be trimodal (there is a mode at each age group)
# Conclusion: Age is trimodal and over-represented for the intermediate age group

# Average Glucose Level Histogram
hist(df$avg_glucose_level, breaks=20, col="lightgreen", main="Avg Glucose Level Distribution with Density", xlab="Avg Glucose Level", freq=FALSE)
lines(density(df$avg_glucose_level, na.rm=TRUE), col="red", lwd=2)
# Average glucose level seems to be bimodal
# Conclusion: We can assume it's bell-shaped but right skewed (needs log scaling)

# BMI Histogram
hist(df$bmi, breaks=20, col="orange", main="BMI Distribution with Density", xlab="BMI", freq=FALSE)
lines(density(df$bmi, na.rm=TRUE), col="red", lwd=2)
# Conclusion: BMI is almost normally distributed, slightly right skewed. May need log scaling
dev.off()

png("plots/pie_stroke.png", width=1000, height=1000)
par(mfrow=c(1,1))

# Stroke Pie Chart
stroke_counts <- table(df$stroke)
pct <- round(stroke_counts / sum(stroke_counts) * 100, 2)
lbls <- paste(c("No Stroke", "Stroke"), pct, sep=" ")
lbls <- paste(lbls, "%", sep="")
pie(stroke_counts, labels=lbls, main="Stroke vs No Stroke Distribution", col=c("lightgreen","red"))
# Conclusion: Data is HIGHLY unbalanced. We need to use SMOTE for data augmentation to balance the dataset for predictions
dev.off()