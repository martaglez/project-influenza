library(readr)
library(dplyr)
library(mgcv)

script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)

csv_path <- file.path(script_dir, "..", "data", "influenza_clean_weekly.csv")

flu <- read_csv(csv_path)

flu$country <- as.factor(flu$country)

flu <- flu %>% arrange(country, year, week)

flu <- flu %>%
  arrange(country, year, week) %>%
  group_by(country) %>%
  mutate(
    lag1 = lag(cases, 1),
    lag2 = lag(cases, 2),
    lag3 = lag(cases, 3),
    lag4 = lag(cases, 4),
    lag5 = lag(cases, 5)) %>%
  ungroup() %>%
  na.omit()

flu <- na.omit(flu)

train <- flu %>%
  group_by(country) %>%
  arrange(year, week) %>%
  mutate(row_id = row_number(), n = n(), split = ifelse(row_id <= 0.8 * n, "train", "test")) %>%
  ungroup() %>%
  filter(split == "train") %>%
  select(-row_id, -n, -split)

test <- flu %>%
  group_by(country) %>%
  arrange(year, week) %>%
  mutate(row_id = row_number(), n = n(), split = ifelse(row_id <= 0.8 * n, "train", "test")) %>%
  ungroup() %>%
  filter(split == "test") %>%
  select(-row_id, -n, -split)

gam_model <- gam(cases ~ s(week, bs = "cc", k = 5) + lag1 + lag2 + lag3 + country,
                 data = train, method = "REML")
summary(gam_model)


pred <- predict(gam_model, newdata = test)

rmse <- sqrt(mean((test$cases - pred)^2))
mae <- mean(abs(test$cases - pred))
r2 <- 1 - sum((test$cases - pred)^2) / sum((test$cases - mean(test$cases))^2)

cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("R2:", r2, "\n")


#PLOTS
library(ggplot2)

plot_data <- test %>%
  arrange(year, week) %>%
  mutate(predicted = predict(gam_model, newdata = test))

plot_data <- plot_data %>% filter(year >= 2022 & year <= 2026)

#Real
p <- ggplot(plot_data) +
  geom_point(aes(x = week, y = cases, color = factor(year)), alpha = 0.6, size = 2) +
  facet_wrap(~ "Real Cases") +
  scale_color_brewer(palette = "Set1") +
  labs(x = "Week", y = "Cases", color = "Year") +
  ylim(0, 400) +
  theme_minimal(base_size = 14)

#Predicted
p2 <- ggplot(plot_data) +
  geom_point(aes(x = week, y = predicted, color = factor(year)), alpha = 0.6, size = 2) +
  facet_wrap(~ "Predicted Cases") +
  scale_color_brewer(palette = "Set1") +
  labs(x = "Week", y = "Cases", color = "Year") +
  ylim(0, 400) +
  theme_minimal(base_size = 14)

library(gridExtra)
grid.arrange(p, p2, ncol = 2)

#ANOTHER PLOT
test_plot <- test %>% filter(year >= 2022 & year <= 2026)

test_plot$pred <- predict(gam_model, newdata = test_plot)

#Real
p_real <- ggplot(test_plot, aes(x = week, y = cases)) +
  geom_point(alpha = 0.6, size = 2, color = "steelblue") +
  facet_wrap(~ year, ncol = 1) +
  ylim(0, 180) +
  labs(title = "Real Cases by Year", x = "Week", y = "Cases") +
  theme_minimal() +
  theme(strip.text = element_text(size = 12, face = "bold"))

#Predicted
p_pred <- ggplot(test_plot, aes(x = week, y = pred)) +
  geom_point(alpha = 0.6, size = 2, color = "firebrick") +
  facet_wrap(~ year, ncol = 1) +
  ylim(0, 180) +
  labs(title = "Predicted Cases by Year", x = "Week", y = "Cases") +
  theme_minimal() +
  theme(strip.text = element_text(size = 12, face = "bold"))

library(patchwork)
p_real | p_pred

