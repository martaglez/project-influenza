library(readr)
library(dplyr)
library(mgcv)
library(ggplot2)

script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)

train_path <- file.path(script_dir, "..", "data", "train.csv")
val_path   <- file.path(script_dir, "..", "data", "val.csv")
test_path  <- file.path(script_dir, "..", "data", "test.csv")

train <- read_csv(train_path)
val   <- read_csv(val_path)
test  <- read_csv(test_path)

train <- train %>% mutate(across(starts_with("country_"), as.numeric))
val   <- val   %>% mutate(across(starts_with("country_"), as.numeric))
test  <- test  %>% mutate(across(starts_with("country_"), as.numeric))

country_vars <- grep("^country_", names(train), value = TRUE)

#GAM MODEL
formula <- as.formula(
  paste(
    "cases ~ s(week, bs='cc', k=5) + lag1 + lag2 + lag3 +",
    paste(country_vars, collapse = " + ")
  )
)

gam_model <- gam(formula, data = train, method = "REML")

summary(gam_model)
plot(gam_model)


#PREDICTIONS
train$pred <- predict(gam_model, newdata = train)
val$pred   <- predict(gam_model, newdata = val)
test$pred  <- predict(gam_model, newdata = test)

#VALIDATION
rmse_val <- sqrt(mean((val$cases - val$pred)^2))
mae_val  <- mean(abs(val$cases - val$pred))
r2_val   <- 1 - sum((val$cases - val$pred)^2) / sum((val$cases - mean(val$cases))^2)

cat("VAL RMSE:", rmse_val, "\n")
cat("VAL MAE :", mae_val, "\n")
cat("VAL R2  :", r2_val, "\n")


#TEST 
rmse_test <- sqrt(mean((test$cases - test$pred)^2))
mae_test  <- mean(abs(test$cases - test$pred))
r2_test   <- 1 - sum((test$cases - test$pred)^2) / sum((test$cases - mean(test$cases))^2)

cat("TEST RMSE:", rmse_test, "\n")
cat("TEST MAE :", mae_test, "\n")
cat("TEST R2  :", r2_test, "\n")

#COUTNRY
get_country <- function(df) {
  country_cols <- grep("^country_", names(df), value = TRUE)
  
  df$country <- apply(df[, country_cols], 1, function(row) {
    idx <- which(row == 1)
    if (length(idx) == 1) {
      return(sub("country_", "", country_cols[idx]))
    } else {
      return(NA)
    }
  })
  
  df$country <- as.factor(df$country)
  return(df)
}

train <- get_country(train)
val   <- get_country(val)
test  <- get_country(test)

data_all <- bind_rows(train, val, test) %>%
  mutate(time = year + week / 52)

#PLOT 1: REAL
ggplot(data_all, aes(x = time, y = cases, color = country, group = country)) +
  geom_line(alpha = 0.7) +
  labs(
    title = "Influenza cases over time by country",
    x = "Time",
    y = "Cases",
    color = "Country"
  ) +
  theme_minimal()

#PLOT 2: predictions
ggplot(data_all, aes(x = time, y = pred, color = country, group = country)) +
  geom_line(alpha = 0.7) +
  labs(
    title = "GAM predictions over time by country",
    x = "Time",
    y = "Predicted cases",
    color = "Country"
  ) +
  theme_minimal()

#PARITY PLOT
test$pred <- predict(gam_model, newdata = test)

ggplot(test, aes(x = cases, y = pred)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Parity Plot (Real vs Predicted)",
    x = "Real Cases",
    y = "Predicted Cases"
  ) +
  theme_minimal()

#SPAIN
country_sel <- "Spain"

df_spain <- data_all %>%
  filter(country_Spain == 1) %>%
  arrange(time)

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

