# MODELO ENDEMIC-EPIDEMIC (hhh4) - INFLUENZA EUROPA
library(readr)
library(dplyr)
library(tidyr)
library(surveillance)
library(rstudioapi)

# DATA
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
data_dir   <- file.path(script_dir, "..", "data")

train <- read_csv(file.path(data_dir, "train.csv"))
val   <- read_csv(file.path(data_dir, "val.csv"))
test  <- read_csv(file.path(data_dir, "test.csv"))

# unir todo
all_data <- bind_rows(
  train %>% mutate(split = "train"),
  val   %>% mutate(split = "val"),
  test  %>% mutate(split = "test")
)

# reconstruir "country"
country_cols <- grep("^country_", names(all_data), value = TRUE)

country_long <- all_data %>%
  mutate(row_id = row_number()) %>%
  select(row_id, all_of(country_cols)) %>%
  pivot_longer(cols = all_of(country_cols),
               names_to  = "country_col",
               values_to = "is_country") %>%
  filter(is_country == TRUE) %>%
  mutate(country = sub("^country_", "", country_col)) %>%
  select(row_id, country)

all_data <- all_data %>%
  mutate(row_id = row_number()) %>%
  left_join(country_long, by = "row_id") %>%
  mutate(country = if_else(is.na(country), "Austria", country)) %>%
  select(-row_id, -all_of(country_cols))

#split: todo antes de 2025 es train, a partir del 2025 es test
all_data <- all_data %>%
  mutate(split = case_when(
    year >= 2025 ~ "test",
    TRUE         ~ "train"
  ))

cat("Split:\n")
print(all_data %>%
        group_by(split) %>%
        summarise(
          min_year  = min(year),
          max_year  = max(year),
          n_semanas = n_distinct(paste(year, week)),
          .groups   = "drop"
        ))

#índice temporal: transformar pares (year, week) en un contador simple t = 1,2...
epoch_order <- all_data %>%
  select(year, week) %>%
  distinct() %>%
  arrange(year, week) %>%
  mutate(t = row_number())

all_data <- all_data %>%
  left_join(epoch_order, by = c("year", "week"))

#matriz wide (semanas (filas) x países (columnas))
obs_matrix <- all_data %>%
  select(t, country, cases) %>%
  pivot_wider(names_from  = country,
              values_from = cases,
              values_fill = 0) %>%
  arrange(t)

time_index  <- obs_matrix$t
obs_matrix  <- obs_matrix %>% select(-t)
countries   <- colnames(obs_matrix)
n_all_rows  <- nrow(obs_matrix)

cat("Países:", length(countries), "\n")
cat("Semanas totales:", n_all_rows, "\n")

#coordenadas y matriz de vecindad
coords <- all_data %>%
  select(country, lat, lon) %>%
  distinct() %>%
  filter(country %in% countries) %>%
  arrange(match(country, countries))

coord_mat           <- as.matrix(coords[, c("lon", "lat")])
rownames(coord_mat) <- coords$country

dist_mat <- as.matrix(dist(coord_mat))

#power-law en vez de exponencial; países cercanos pesan más; cae es más gradual
W <- (dist_mat + 1)^(-1)
diag(W) <- 0
W <- W / rowSums(W)

#OBJETO STS (surveillance time series) COMPLETO
#creamos un dataframe con solo las columnas t, year y week
epoch_df <- all_data %>%
  select(t, year, week) %>%
  distinct() %>%
  arrange(t)

#convertimos año + semana -> fecha real, construimos strings tipo: "2024-W03-1"
epoch_dates <- as.Date(
  paste0(epoch_df$year, "-W", sprintf("%02d", epoch_df$week), "-1"),
  format = "%Y-W%W-%u"
)

#sts
sts_all <- sts( 
  observed      = as.matrix(obs_matrix),
  epoch         = epoch_dates,
  epochAsDate   = TRUE,
  frequency     = 52,
  neighbourhood = W
)

#índices train/test
t_train <- unique(all_data$t[all_data$split == "train"])
t_test  <- unique(all_data$t[all_data$split == "test"])

row_train <- which(time_index %in% t_train)
row_test  <- which(time_index %in% t_test)

n_train  <- length(row_train)
sts_test <- sts_all[row_test, ]

cat("Semanas train:", n_train, "\n")
cat("Semanas test:",  length(row_test), "\n")

#MODELO hhh4 !!!!!!
control_all <- list(
  
  #componente endémica
  end = list(
    f = addSeason2formula(
      # intercepto distinto por país (unit-specific)
      # -1 = quitamos intercepto global
      # fe(1, unitSpecific=TRUE) = cada país tiene su propio baseline
      ~ -1 + fe(1, unitSpecific = TRUE),  
      
      period  = 52,
      timevar = "t",
      
      # número de armónicos de Fourier
      # S = 6 → modelo flexible de estacionalidad (más picos)
      S = 6                          
    )
  ),
  
  #componente autorregresiva
  ar = list(
    f = addSeason2formula(
      #dependencia del pasado (casos previos )
      ~ 1,
    
      period = 52, timevar = "t",
      
      #S = 1 --> estacionalidad simple en AR
      #AR también con estacionalidad: captura que en invierno cada país depende
      #más de su semana anterior
      S = 1
    )
  ),
  
  #componente espacial
  ne = list(
    f       = ~ 1, #efecto de contagio entre regiones
    weights = neighbourhood(sts_all) #matriz de vecindad
  ),
  
  family = "NegBin1", #distribución: binomial negativa (sobredispersión)
  data   = list(t = seq_len(n_all_rows)) #índice temporal
)

cat("Ajustando modelo hhh4 mejorado...\n")
modelo_all <- hhh4(sts_all, control = control_all)
summary(modelo_all)

#PREDICTION ONE-STEP-AHEAD
cat("Calculando predicciones...\n")

osa <- oneStepAhead(
  #modelo ya entrenado con TODO (sts_all)
  modelo_all,
  
  #tp = tiempos donde se evalúa la predicción
  #n_train → último punto de entrenamiento
  #n_all_rows - 1 → penúltima observación total
  tp = c(n_train, n_all_rows - 1),
  
  #usamos predicción final (no aproximaciones internas)
  type = "final"
)

pred_osa <- osa$pred
cat("dim pred_osa:", dim(pred_osa), "\n")

#metrics
obs_test <- observed(sts_test)

n_rows   <- min(nrow(pred_osa), nrow(obs_test))
pred_osa <- pred_osa[1:n_rows, ]
obs_eval <- obs_test[1:n_rows, ]

common_countries <- intersect(colnames(obs_eval), colnames(pred_osa))
obs_eval <- obs_eval[, common_countries]
pred_osa <- pred_osa[, common_countries]

obs_eval <- matrix(as.numeric(obs_eval), nrow = n_rows,
                   dimnames = list(NULL, common_countries))
pred_osa <- matrix(as.numeric(pred_osa), nrow = n_rows,
                   dimnames = list(NULL, common_countries))

mae_by_country  <- colMeans(abs(obs_eval - pred_osa), na.rm = TRUE)
rmse_by_country <- sqrt(colMeans((obs_eval - pred_osa)^2, na.rm = TRUE))
ss_res          <- colSums((obs_eval - pred_osa)^2, na.rm = TRUE)
ss_tot          <- colSums((obs_eval - colMeans(obs_eval))^2, na.rm = TRUE)
r2_by_country   <- 1 - ss_res / ss_tot

results <- data.frame(
  country = common_countries,
  MAE     = round(mae_by_country,  2),
  RMSE    = round(rmse_by_country, 2),
  R2      = round(r2_by_country,   4)
) %>% arrange(desc(R2))

print(results)
cat("\nMAE global:",  round(mean(mae_by_country),  2), "\n")
cat("RMSE global:", round(mean(rmse_by_country), 2), "\n")

cat("R2 global:",   round(mean(r2_by_country),   4), "\n")

#--------------PLOTS--------------------------------
plot(modelo_all, type = "fitted", total = TRUE,
     col = c("tomato", "steelblue", "seagreen3"))

plot_country <- function(pais) {
  idx      <- which(common_countries == pais)
  obs_vec  <- obs_eval[, idx]
  pred_vec <- pred_osa[, idx]
  ylim_max <- max(c(obs_vec, pred_vec), na.rm = TRUE)
  
  plot(obs_vec, type = "l", col = "black", lwd = 2,
       ylim = c(0, ylim_max),
       ylab = "Casos", xlab = "Semana",
       main = paste(pais, "- Observado vs Predicho (test)"))
  lines(pred_vec, col = "red", lwd = 2, lty = 2)
  legend("topright",
         legend = c("Observado", "Predicho"),
         col    = c("black", "red"),
         lty    = c(1, 2), lwd = 2)
}

plot_country("Spain")
plot_country("Germany")
plot_country("France")


library(ggplot2)
library(tidyr)
library(dplyr)

# DATOS LARGOS PARA GGPLOT
# Reconstruir eje temporal como año + semana/52 
epoch_test <- epoch_df[row_test, ]
time_test  <- epoch_test$year + epoch_test$week / 52

obs_df  <- as.data.frame(obs_eval)
pred_df <- as.data.frame(pred_osa)

obs_df$time  <- time_test[1:n_rows]
pred_df$time <- time_test[1:n_rows]

obs_long <- obs_df %>%
  pivot_longer(-time, names_to = "country", values_to = "cases")

pred_long <- pred_df %>%
  pivot_longer(-time, names_to = "country", values_to = "predicted")

combined <- left_join(obs_long, pred_long, by = c("time", "country"))

#PARITY PLOT
parity_df <- data.frame(
  real      = as.vector(obs_eval),
  predicted = as.vector(pred_osa)
)

ggplot(parity_df, aes(x = real, y = predicted)) +
  geom_point(alpha = 0.3, size = 1.5, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  labs(title = "Parity plot (Real vs Predicted)",
       x = "Real cases", y = "Predicted cases") +
  theme_minimal()

#SPAIN: real vs predictions
esp <- combined %>% filter(country == "Spain")

ggplot(esp, aes(x = time)) +
  geom_line(aes(y = cases,     color = "Real"),      linewidth = 1, alpha = 0.8) +
  geom_line(aes(y = predicted, color = "Predicted"), linewidth = 1, alpha = 0.8) +
  scale_color_manual(values = c("Real" = "black", "Predicted" = "red")) +
  labs(title = "Real vs Predicted influenza cases - Spain (test)",
       x = "Year", y = "Cases", color = "") +
  theme_minimal()

#TRAIN + TEST CON INTERVALO DE CONFIANZA
epoch_train <- epoch_df[row_train, ]
time_train  <- epoch_train$year + epoch_train$week / 52

obs_train_full <- observed(sts_all[row_train, ])
esp_train_obs  <- obs_train_full[, "Spain"]

fitted_train   <- fitted(modelo_all)
esp_train_pred <- fitted_train[1:length(time_train), "Spain"]

# psi: parámetro de sobredispersión
psi <- exp(coef(modelo_all)["overdisp"])

# Varianza NegBin1: var = mu + mu²/psi — tanto para train como test
esp_train_var <- as.numeric(esp_train_pred) + as.numeric(esp_train_pred)^2 / psi

esp_train_sd  <- sqrt(esp_train_var)

esp_test_pred <- pred_osa[, "Spain"]
esp_test_var  <- as.numeric(esp_test_pred) + as.numeric(esp_test_pred)^2 / psi
esp_test_sd   <- sqrt(esp_test_var)

df_train <- data.frame(
  time      = time_train,
  actual    = as.numeric(esp_train_obs),
  predicted = as.numeric(esp_train_pred),
  sd        = as.numeric(esp_train_sd),
  split     = "Train"
)

df_test <- data.frame(
  time      = time_test[1:n_rows],
  actual    = as.numeric(obs_eval[, "Spain"]),
  predicted = as.numeric(esp_test_pred),
  sd        = as.numeric(esp_test_sd),
  split     = "Test"
)

df_all <- bind_rows(df_train, df_test)

ggplot(df_all, aes(x = time)) +
  geom_ribbon(aes(ymin = predicted - 1.96 * sd,
                  ymax = predicted + 1.96 * sd,
                  fill = split),
              alpha = 0.2) +
  geom_point(aes(y = actual,    color = split), alpha = 0.5, size = 1) +
  geom_line(aes(y  = predicted, color = split), linewidth = 0.8) +
  scale_color_manual(values = c("Train" = "blue", "Test" = "red")) +
  scale_fill_manual(values  = c("Train" = "blue", "Test" = "red")) +
  labs(title = "hhh4 Model: Actual vs Predicted - Spain",
       x = "Year", y = "Influenza Cases", color = "", fill = "") +
  theme_minimal() +
  theme(legend.position = "top")

