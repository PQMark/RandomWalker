# Load the jsonlite package
library(jsonlite)
library(writexl)
library(dplyr)

# Define the path to your JSON file
json_file_path <- "results.json"

# Read the JSON file
json_data <- fromJSON(json_file_path)
feature_names <- unlist(json_data[[1]])

# Now, use cat() to display the vector
cat(feature_names, sep = "\n")

selected_rows <- json_data %>% filter(Features_Count <= 7)

# View the selected rows
print(selected_rows)

json_data$Features_Count <- sapply(json_data$Features, length)

#json_data$delta <- json_data$AvgF1 - json_data$ErrorF1


data <- data.frame(NumFeatures = json_data$Features_Count,  AvgF1 = json_data$AvgF1, ErrorF1 = json_data$ErrorF1)
data <- data[data$NumFeatures >= 1 & data$NumFeatures <= 200, ]

library(ggplot2)

scale_factor <- 1
p <- ggplot(data, aes(x = NumFeatures, y = AvgF1)) +   
  geom_line(colour = "blue") +   
  geom_errorbar(
    aes(ymin = AvgF1 - ErrorF1*scale_factor, ymax = AvgF1 + ErrorF1*scale_factor), 
    width = 0.2, 
    colour = "blue"
  ) +   
  labs(
    title = "Recursive Feature Elimination with correlated features",
    x = "Number of Features Selected",
    y = "Average F1-score"
  ) +   
  scale_x_continuous(breaks = seq(1, 200, by = 20)) +   
  scale_y_continuous(breaks = seq(0.85, 1, by = 0.1), limits = c(0.85, 1)) +   
  theme_minimal()

# Display the plot
print(p)
