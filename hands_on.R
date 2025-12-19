library(forecast)
library(tseries)
library(readr)
library(zoo) 

#  Step 1: Load Data and Visualize the Time Series

df <- read_csv("/Users/maiab/Downloads/monthly_sales.csv")

ts_data <- ts(df$sales, start = c(2013,1), frequency = 12)
print(ts_data)

plot(ts_data, type="l", col="blue",
     main=paste("Monthly Sales Time Series"),
     ylab="Value")

# Step 2
stationarize_series <- function(series, title_text="Time Series") {
  cat("\n=====================================\n")
  cat("Checking stationarity for:", title_text, "\n")
  cat("=====================================\n")
  
  roll_mean <- rollmean(series, k = 12, fill = NA)
  roll_sd <- rollapply(series, width = 12, FUN = sd, fill = NA)
  
  # Plot
  try({plot(series, type="l", col="blue",
       main=paste("Rolling Mean & SD -", title_text),
       ylab="Value")
  lines(roll_mean, col="red", lwd=2)
  lines(roll_sd, col="green", lwd=2)
  legend("topleft",
         legend=c("Original", "Rolling Mean", "Rolling SD"),
         col=c("blue", "red", "green"), lty=1)
})
  
  # ADF Test
  clean_series <- na.omit(series)
  adf <- adf.test(clean_series)
  cat("ADF Statistic:", adf$statistic, "\n")
  cat("p-value:", adf$p.value, "\n")
  print(adf)
  
}

# Step 2

stationarize_series(ts_data, "Original Series") 

simple_diff <- diff(ts_data, lag=1)
stationarize_series(simple_diff, "First Difference (Lag 1)") 

seasonal_diff <- diff(simple_diff, lag=12) 
stationarize_series(seasonal_diff, "Seasonal Difference (Lag 12) on First Diff")

# step 3
acf(seasonal_diff, main="ACF - Fully Stationary Series")
pacf(seasonal_diff, main="PACF - Fully Stationary Series")

model <- Arima(ts_data, 
               order=c(0, 1, 1), 
               seasonal=c(0, 1, 1), 
               method="ML") 

summary(model)

# Step 5: Predict
future <- forecast(model, h=60)

plot(future,
     main="Monthly Sales Forecast (5 Years)",
     ylab="Sales") 

plot(ts_data, col="blue", main="Actual vs Fitted & Forecast", lwd=2)
lines(fitted(model), col="green", lwd=2)
lines(future$mean, col="red", lwd=2)
legend("topleft",
       legend=c("Actual", "Fitted", "Forecast"),
       col=c("blue", "green", "red"), lty=1)