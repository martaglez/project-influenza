library(readr)
library(dplyr)
library(mgcv)

flu <- read_csv("influenza_clean_weekly.csv")

flu$country <- as.factor(flu$country)

flu <- flu %>% arrange(country, year, week)

flu <- flu %>% group_by(country) %>% mutate(
    lag1 = lag(cases, 1),
    lag2 = lag(cases, 2),
    lag3 = lag(cases, 3),
    lag4 = lag(cases, 4),
    lag5 = lag(cases, 5)) %>% ungroup()

flu <- na.omit(flu)

gam_model <- gam(cases ~ 
    s(week, bs = "cc", k = 5) +
    s(lag1) + 
    s(lag2) + 
    s(lag3) +
    s(lag4) +
    s(lag5) +
    country, data = flu, method = "REML")

summary(gam_model)

plot(gam_model, pages = 1, shade = TRUE, rug = TRUE)

pred <- predict(gam_model, newdata = flu)

rmse <- sqrt(mean((flu$cases - pred)^2))
mae <- mean(abs(flu$cases - pred))
r2 <- 1 - sum((flu$cases - pred)^2) / sum((flu$cases - mean(flu$cases))^2)

cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("R2:", r2, "\n")

spain_data <- flu %>% filter(country == "Spain") %>% arrange(year, week)

last5 <- tail(spain_data$cases, 5)

new_data <- data.frame(
  country = factor("Spain", levels = levels(flu$country)),
  week = 1,
  lag1 = last5[5],
  lag2 = last5[4],
  lag3 = last5[3],
  lag4 = last5[2],
  lag5 = last5[1])

pred <- predict(gam_model, newdata = new_data)

cat("Predicted cases Spain 2023-W01:", round(pred), "\n")