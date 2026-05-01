library(readr)
library(dplyr)
library(mgcv)

script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
data_dir <- file.path(script_dir, "..", "data")

train <- read_csv(file.path(data_dir, "train.csv"))
val   <- read_csv(file.path(data_dir, "val.csv"))
test  <- read_csv(file.path(data_dir, "test.csv"))

train$country <- as.factor(train$country)
val$country   <- as.factor(val$country)
test$country  <- as.factor(test$country)


# GAM
gam_model <- gam(
  cases ~ 
    s(year, k = 3) +
    s(week, bs = "cc", k = 3),
  data = train,
  method = "REML"
)

summary(gam_model)
plot(gam_model)

pred_val <- predict(gam_model, newdata = val)

rmse_val <- sqrt(mean((val$cases - pred_val)^2))
mae_val  <- mean(abs(val$cases - pred_val))
r2_val   <- 1 - sum((val$cases - pred_val)^2) / sum((val$cases - mean(val$cases))^2)

cat("Validation RMSE:", rmse_val, "\n")
cat("Validation MAE:", mae_val, "\n")
cat("Validation R2:", r2_val, "\n")

pred_test <- predict(gam_model, newdata = test)

rmse <- sqrt(mean((test$cases - pred_test)^2))
mae  <- mean(abs(test$cases - pred_test))
r2   <- 1 - sum((test$cases - pred_test)^2) / sum((test$cases - mean(test$cases))^2)

cat("Test RMSE:", rmse, "\n")
cat("Test MAE:", mae, "\n")
cat("Test R2:", r2, "\n")


#PLOTS
#Real
library(ggplot2)
library(dplyr)

data_all <- bind_rows(train, val, test)

data_all <- data_all %>%
  mutate(time = year + week/52)

ggplot(data_all, aes(x = time, y = cases, color = country)) +
  geom_line(alpha = 0.7) +
  labs(
    title = "Influenza Cases Over Time by Country",
    x = "Year",
    y = "Cases"
  ) +
  theme_minimal() +
  theme(legend.position = "right")

#Prediction
data_all <- data_all %>%
  mutate(pred = predict(gam_model, newdata = data_all))

ggplot(data_all, aes(x = time, y = pred, color = country)) +
  geom_line(alpha = 0.4) +
  labs(
    title = "Model Predictions Across Countries (GAM)",
    x = "Year",
    y = "Predicted Cases"
  ) +
  theme_minimal() +
  theme(legend.position = "right")

#Real vs predicted
pred_test <- predict(gam_model, newdata = test)

ggplot(test, aes(x = cases, y = pred_test)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Parity Plot (Real vs Predicted)",
    x = "Real Cases",
    y = "Predicted Cases"
  ) +
  theme_minimal()

#Spain
country <- "Spain"

df_country <- data_all %>%
  filter(country == !!country) %>%
  arrange(year, week)

df_country <- df_country %>%
  mutate(pred = predict(gam_model, newdata = df_country),
         time = year + week/52)

ggplot(df_country, aes(x = time)) +
  geom_line(aes(y = cases, color = "Real"), alpha = 0.7) +
  geom_line(aes(y = pred, color = "Predicted"), alpha = 0.7) +
  labs(
    title = paste("Real vs Predicted Influenza Cases -", country),
    x = "Year",
    y = "Cases",
    color = ""
  ) +
  theme_minimal()

library(patchwork)
geom_line(real)
geom_line(pred)

