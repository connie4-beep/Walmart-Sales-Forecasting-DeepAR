# ==========================================
# SETUP AND DATA LOADING
# ==========================================
gc()
# setwd("/Users/conniezhang/Desktop/ECON491/data/") 
# (Uncomment setwd when running on your machine)

# install.packages(c("pdftools", "data.table", "tidyverse", "lubridate", "stargazer"))
library(data.table)
library(readr)
# library(pdftools) # Only needed if extracting from PDF
library(tidyverse)
library(lubridate)
library(stargazer)

cat("Loading data...\n")
sales_test_validation_data <- read_csv("Sales Test Validation.csv")
sales_train_validation_data <- read_csv("Sales Train Validation.csv")
calendar_data <- read_csv("Calendar (1).csv")
prices <- fread("sell_prices.csv") 

# 1. Clear everything to free up maximum RAM (optional if starting fresh)
# rm(list = ls())
# gc()

# ==========================================
# 2. COMBINE AND MELT SALES DATA
# ==========================================
cat("Melting sales data...\n")
full_sales <- left_join(
  sales_train_validation_data,
  sales_test_validation_data,
  by = c("item_id", "dept_id", "cat_id", "store_id", "state_id")
)

rm(sales_train_validation_data, sales_test_validation_data)
gc()

setDT(full_sales)
full_sales[, c("cat_id", "state_id") := NULL]

long_sales <- melt(full_sales, 
                   id.vars = c("item_id", "dept_id", "store_id"), 
                   measure.vars = patterns("^d_"), 
                   variable.name = "d", 
                   value.name = "sales")

rm(full_sales)
gc()

# ==========================================
# 3. PREP CALENDAR AND JOIN
# ==========================================
cat("Joining calendar...\n")
setDT(calendar_data)
calendar_data[, d := paste0("d_", 1:.N)] 
slim_calendar <- calendar_data[, .(d, date, wm_yr_wk, wday, month, event_name_1, event_type_1)]

final_df <- slim_calendar[long_sales, on = "d"]

final_df <- as.data.frame(final_df) %>%
  mutate(
    sales = as.numeric(sales),
    date = as.Date(date)
  )

rm(long_sales, slim_calendar)
gc()

# ==========================================
# 4. REBUILD SALES-WEIGHTED PRICES
# ==========================================
cat("Building weighted prices...\n")
# Aggregate lifetime sales for weights
item_weights <- final_df %>%
  group_by(item_id, store_id, dept_id) %>%
  summarize(item_lifetime_sales = sum(sales, na.rm = TRUE), .groups = "drop")

prices_mapped <- merge(prices, item_weights, by = c("item_id", "store_id"))

dept_prices <- prices_mapped %>%
  group_by(store_id, dept_id, wm_yr_wk) %>%
  summarize(avg_price = sum(sell_price * item_lifetime_sales, na.rm = TRUE) / 
              sum(item_lifetime_sales, na.rm = TRUE), .groups = "drop")

rm(prices, prices_mapped, item_weights)
gc()

# ==========================================
# 5. ASSEMBLE FINAL MODEL DATA
# ==========================================
cat("Assembling final model data...\n")

# Aggregate daily sales up to the store-department level so it matches our prices
model_data <- final_df %>%
  group_by(store_id, dept_id, date, wm_yr_wk, wday, month, event_type_1) %>%
  summarize(total_sales = sum(sales, na.rm = TRUE), .groups = "drop")

# Join the prices
model_data <- left_join(model_data, dept_prices, by = c("store_id", "dept_id", "wm_yr_wk"))

# Create the specific Dummy Variables required for the benchmark model
model_data <- model_data %>%
  mutate(
    day_of_week = as.factor(wday),
    month_factor = as.factor(month),
    is_holiday = ifelse(is.na(event_type_1), 0, 1) # 1 if there is an event, 0 if normal day
  )

# Remove NA prices just in case
model_data <- model_data %>% filter(!is.na(avg_price))

# ==========================================
# 6. TRAIN/TEST SPLIT
# ==========================================
cat("Splitting data into Train and Test...\n")

# The training data ends on Day 1913 (2016-04-24)
train_data <- model_data %>% filter(date <= as.Date("2016-04-24"))
test_data <- model_data %>% filter(date > as.Date("2016-04-24"))


# ==========================================
# 7. RUN THE SIMPLE BENCHMARK MODEL
# ==========================================
cat("Running the Benchmark OLS Regression...\n")

# The Simple Benchmark OLS Model (No weights, no lags)
simple_model <- lm(total_sales ~ avg_price + day_of_week + month_factor + is_holiday, 
                   data = train_data)

# Output results using stargazer (as requested in the syllabus)
stargazer(simple_model, type = "text", title = "Simple Benchmark Model Results")


# ==========================================
# 8. PREDICTION & ACCURACY EVALUATION
# ==========================================
cat("Predicting and evaluating...\n")

# Predict the sales for the 28-day test period
test_data$predicted_sales <- predict(simple_model, newdata = test_data)

# Aggregate the test data to daily totals for checking accuracy
daily_accuracy <- test_data %>%
  group_by(date) %>%
  summarize(
    Actual = sum(total_sales, na.rm = TRUE),
    Predicted = sum(predicted_sales, na.rm = TRUE),
    .groups = "drop"
  )

# Calculate RMSE and MAPE
rmse_value <- sqrt(mean((daily_accuracy$Actual - daily_accuracy$Predicted)^2))
mape_value <- mean(abs((daily_accuracy$Actual - daily_accuracy$Predicted) / daily_accuracy$Actual)) * 100

cat("-----------------------------------\n")
cat("Daily Total Sales RMSE:", round(rmse_value, 2), "units\n")
cat("Daily Total Sales MAPE:", round(mape_value, 2), "%\n")
cat("-----------------------------------\n")

# ==========================================
# 9. VISUALIZE PREDICTED VS ACTUAL
# ==========================================
comparison_data <- daily_accuracy %>%
  pivot_longer(cols = c(Actual, Predicted), names_to = "Type", values_to = "Sales")

comparison_plot <- ggplot(comparison_data, aes(x = date, y = Sales, color = Type, linetype = Type)) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = c("Actual" = "#2c3e50", "Predicted" = "#e74c3c")) + 
  theme_minimal() +
  labs(
    title = "Simple Benchmark Model: Predicted vs Actual Sales",
    subtitle = "Test Validation Period (Days 1914 - 1941)",
    x = "Date",
    y = "Total Units Sold",
    color = "Metric",
    linetype = "Metric"
  ) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )

print(comparison_plot)