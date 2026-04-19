library(readr)
library(dplyr)
library(mgcv)

script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)

train_path <- file.path(script_dir, "..", "data", "train.csv")
val_path   <- file.path(script_dir, "..", "data", "val.csv")
test_path  <- file.path(script_dir, "..", "data", "test.csv")

train <- read_csv(train_path)
val   <- read_csv(val_path)
test  <- read_csv(test_path)

train$country <- as.factor(train$country)
val$country   <- as.factor(val$country)
test$country  <- as.factor(test$country)

gam_model <- gam(
  cases ~ s(week, bs = "cc", k = 5) + lag1 + lag2 + lag3 + country, #country, no lat and lon
  data = train,
  method = "REML")

summary(gam_model)

val_pred <- predict(gam_model, newdata = val)

rmse_val <- sqrt(mean((val$cases - val_pred)^2))
mae_val  <- mean(abs(val$cases - val_pred))
r2_val   <- 1 - sum((val$cases - val_pred)^2) / sum((val$cases - mean(val$cases))^2)

cat("VAL RMSE:", rmse_val, "\n")
cat("VAL MAE:", mae_val, "\n")
cat("VAL R2:", r2_val, "\n")

test_pred <- predict(gam_model, newdata = test)

rmse_test <- sqrt(mean((test$cases - test_pred)^2))
mae_test  <- mean(abs(test$cases - test_pred))
r2_test   <- 1 - sum((test$cases - test_pred)^2) / sum((test$cases - mean(test$cases))^2)

cat("TEST RMSE:", rmse_test, "\n")
cat("TEST MAE:", mae_test, "\n")
cat("TEST R2:", r2_test, "\n")


# PLOTS (GAM MODEL)
library(ggplot2)
library(dplyr)

train$pred <- predict(gam_model, newdata = train)
val$pred   <- predict(gam_model, newdata = val)
test$pred  <- predict(gam_model, newdata = test)

train$set <- "train"
val$set   <- "val"
test$set  <- "test"

data_all <- bind_rows(train, val, test) %>%
  mutate(time = year + week / 52)


# REAL
ggplot(data_all, aes(x = time, y = cases, color = country)) +
  geom_line(alpha = 0.7) +
  labs(
    title = "Influenza cases over time by country",
    x = "Time",
    y = "Cases",
    color = "Country"
  ) +
  theme_minimal()


# PREDICTIONS 
ggplot(data_all, aes(x = time, y = pred, color = country)) +
  geom_line(alpha = 0.7) +
  labs(
    title = "GAM predictions over time by country",
    x = "Time",
    y = "Predicted cases",
    color = "Country"
  ) +
  theme_minimal()


# PARITY PLOT
test$pred <- predict(gam_model, newdata = test)

library(ggplot2)

ggplot(test, aes(x = cases, y = pred)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Parity Plot (Real vs Predicted)",
    x = "Real Cases",
    y = "Predicted Cases"
  ) +
  theme_minimal()


# SPAIN (REAL VS PRED)
country_sel <- "Spain"

df_spain <- data_all %>%
  filter(country == country_sel) %>%
  arrange(year, week)

ggplot(df_spain, aes(x = time)) +
  geom_line(aes(y = cases, color = "Real"), linewidth = 0.8) +
  geom_line(aes(y = pred, color = "Predicted"), linewidth = 0.8, alpha = 0.7) +
  labs(
    title = "Spain: Real vs Predicted influenza cases",
    x = "Time",
    y = "Cases",
    color = ""
  ) +
  theme_minimal()
