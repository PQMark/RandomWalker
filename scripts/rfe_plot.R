# Load the jsonlite package
library(jsonlite)

# Define the path to your JSON file
json_file_path <- "results.json"

# Read the JSON file
json_data <- fromJSON(json_file_path)

json_data$Features_Count <- sapply(json_data$Features, length)



data <- data.frame(NumFeatures = json_data$Features_Count,  AvgF1 = json_data$AvgF1, ErrorF1 = json_data$ErrorF1)
data <- data[data$NumFeatures >= 1 & data$NumFeatures <= 20, ]

library(ggplot2)


p <- ggplot(data, aes(x = NumFeatures, y = AvgF1)) +   
  geom_line(colour = "blue") +   
  geom_errorbar(
    aes(ymin = AvgF1 - ErrorF1, ymax = AvgF1 + ErrorF1), 
    width = 0.2, 
    colour = "blue"
  ) +   
  labs(
    title = "Recursive Feature Elimination with correlated features",
    x = "Number of Features Selected",
    y = "Mean Test Accuracy"
  ) +   
  scale_x_continuous(breaks = seq(1, 21, by = 4)) +   
  scale_y_continuous(breaks = seq(0.7, 1, by = 0.1), limits = c(0.7, 1)) +   
  theme_minimal()

# Display the plot
print(p)
