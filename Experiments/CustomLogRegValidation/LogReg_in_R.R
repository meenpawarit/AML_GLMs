

data <- read.csv('Experiments/Data/processed_cleveland_r.csv')
summary(data)
#  Load packages 
library(MASS)

# Ensure categorical variables are factors
categorical_cols <- c("cp", "sex", "fbs", "restecg", "exang", "slope", "thal")

for (col in categorical_cols) {
  data[[col]] <- factor(data[[col]])
}

# Ensure the target is binary (0 vs 1)
if (any(data$target > 1)) {
  data$target <- ifelse(data$target == 0, 0, 1)
  data$target <- factor(data$target)
}

# Fit the full logistic regression model
full_model <- glm(
  target ~ age + sex + cp + trestbps + chol + fbs + restecg +
    thalach + exang + oldpeak + slope + ca + thal,
  family = binomial,
  data = data
)

# Review the model
summary(full_model)

#  Stepwise selection (AIC) 
step_model <- step(full_model, direction = "backward", trace = TRUE)

# Final selected model
summary(step_model)

# Extract coefficients for comparison with your Python implementation
coef(step_model)
