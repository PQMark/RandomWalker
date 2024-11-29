# Load the jsonlite package
library(jsonlite)

# Define the path to your JSON file
json_file_path <- "results.json"

# Read the JSON file
json_data <- fromJSON(json_file_path)

json_data$Features_Count <- sapply(json_data$Features, length)

json_data$delta <- json_data$AvgF1 - json_data$ErrorF1


data <- data.frame(NumFeatures = json_data$Features_Count, DeltaValues = json_data$delta)
data <- data[data$NumFeatures >= 1 & data$NumFeatures <= 20, ]

library(ggplot2)

p <- ggplot(data, aes(x = NumFeatures, y = DeltaValues)) +   
  geom_line(colour = "blue") +   
  #geom_errorbar(aes(ymin = DeltaValues - ErrorValues, ymax = DeltaValues + ErrorValues), width = 0.2, colour = "blue") +   
  labs(
    title = "Recursive Feature Elimination with correlated features",
    x = "Number of Features Selected",
    y = "Mean Test Accuracy"
  ) +   
  scale_x_continuous(breaks = seq(1, 21, by = 4)) +   
  scale_y_continuous(breaks = seq(0.4, 1, by = 0.1), limits = c(0.4, 1)) +   
  theme_minimal()

# Display the plot
print(p)
