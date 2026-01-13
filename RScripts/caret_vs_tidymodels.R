pacman::p_load(
  conflicted,
  tidyverse,
  wrappedtools, # just tools
  palmerpenguins, # data
  ggforce, # for cluster plots, hulls, zoom etc
  ggbeeswarm,
  flextable,
  caret, # Classification and Regression Training
  tidymodels, # modeling framework #
  gmodels, # tools for model fitting
  kknn, # KNN engine for tidymodels
  yardstick, # model performance
  waldo # for result comparisons
)
# conflict_scout()
conflicts_prefer(
  dplyr::slice,
  dplyr::filter,
  palmerpenguins::penguins
)

penguins <- penguins |>
  drop_na() |>
  mutate(ID = paste("P", row_number()) |>
    factor()) |>
  select(ID, species, contains("_"))

predvars <- ColSeeker(penguins, "_")

predvars$names

# preprocessing ####
## caret ####
scaled <- penguins |>
  select(predvars$names) |>
  caret::preProcess(method = c("center", "scale"))
penguins_scaled_caret <- predict(scaled, penguins)

ranged <- penguins |>
  select(predvars$names) |>
  caret::preProcess(method = c("range"))
penguins_ranged_caret <- predict(ranged, penguins)


## tidymodels ####

# Build recipe for scaling
rec_scaled <- recipe(~., data = penguins) |>
  step_center(all_of(predvars$names)) |>
  step_scale(all_of(predvars$names))

# Fit recipe and apply to data
penguins_scaled_tidy <- rec_scaled |>
  prep() |>
  bake(new_data = penguins)

rec_ranged <- recipe(~., data = penguins) |>
  step_range(all_of(predvars$names), min = 0, max = 1)

penguins_ranged_tidy <- rec_ranged |>
  prep() |>
  bake(new_data = penguins)

## comparisons ####

compare(penguins_scaled_caret, penguins_scaled_tidy)

compare(penguins_ranged_caret, penguins_ranged_tidy)


# Data split ####
## caret ####
set.seed(2026)
trainindex <- createDataPartition(
  y = penguins_scaled_caret$species,
  times = 1,
  p = 0.75,
  list = FALSE
)

traindata_caret <- penguins_scaled_caret |> slice(trainindex[, 1])
testdata_caret <- penguins_scaled_caret |> slice(-trainindex[, 1])

## tidymodels ####
set.seed(2026)
data_split <- initial_split(
  penguins_scaled_tidy,
  prop = 0.75,
  strata = species
)
traindata_tidy <- training(data_split)
testdata_tidy <- testing(data_split)


# modelling defaults ####
## caret ####
ctrl <- trainControl(
  method = "repeatedcv",
  number = 5, # 5-fold cross-validation
  repeats = 25
) # 25 repeats

## tidymodels ####
cv_folds <- vfold_cv(
  penguins,
  v = 5,
  repeats = 25
)

# Specific tuning parameters ####
## caret ####
tune_knn <- expand.grid(k = seq(1, 9, 2))

## tidymodels ####
# Define the KNN model and specify 'neighbors' should be tuned
knn_spec <- nearest_neighbor(neighbors = tune()) |>
  set_engine("kknn") |>
  set_mode("classification")
# Define the grid: k = 1, 3, 5, 7, 9
# This replaces expand.grid(k = seq(1, 9, 2))
knn_grid <- grid_regular(
  neighbors(range = c(1, 9)),
  levels = 5
)

# Model training/tuning ####
## caret ####
knnfit <- train(
  form = species ~ .,
  data = traindata_caret,
  method = "knn",
  metric = "Accuracy",
  trControl = ctrl,
  tuneGrid = tune_knn
)
knnfit[["bestTune"]]

## tidymodels ####
# Build workflow
knn_workflow <- workflow() |>
  add_model(knn_spec) |>
  add_formula(species ~ .)
# Tune the model
knn_tuned <- tune_grid(
  knn_workflow,
  resamples = cv_folds,
  grid = knn_grid,
  metrics = metric_set(accuracy, kap),
  control = control_grid(parallel_over = "resamples")
)
knn_tuned |>
  collect_metrics() |>
  filter(.metric == "accuracy") |>
  arrange(desc(mean)) |>
  slice(1)

# Final model evaluation ####
## caret ####
knnfit$resample |>
  pivot_longer(-Resample,
    names_to = "Measure",
    values_to = "Value"
  ) |>
  ggplot(aes(Measure, Value)) +
  geom_boxplot(outlier.alpha = 0) +
  geom_beeswarm(alpha = .3, cex = .25)

## tidymodels ####
best_k <- knn_tuned |>
  select_best(metric = "accuracy")
knn_tuned |>
  collect_metrics(summarize = FALSE) |>
  filter(neighbors == best_k$neighbors) |>
  ggplot(aes(.metric, .estimate)) +
  geom_boxplot(outlier.alpha = 0) +
  geom_beeswarm(alpha = .3, cex = .25)
