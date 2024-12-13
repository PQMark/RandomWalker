# Load the jsonlite package
library(jsonlite)

# Define the path to your JSON file
json_file_path <- "temp/results_rfe_fold_1.json"


# Read the JSON file
json_data <- fromJSON(json_file_path)

json_data$Features_Count <- sapply(json_data$Features, length)



data <- data.frame(NumFeatures = json_data$Features_Count,  AvgF1 = json_data$AvgF1, ErrorF1 = json_data$ErrorF1)
data <- data[data$NumFeatures >= 1 & data$NumFeatures <= max(data$NumFeatures), ]


library(ggplot2)

# Min and Max NumFeatures
x_min <- min(data$NumFeatures, na.rm = TRUE)  
x_max <- max(data$NumFeatures, na.rm = TRUE) 
x_breaks <- seq(x_min, x_max, by = round((x_max - x_min) / 5))

p <- ggplot(data, aes(x = NumFeatures, y = AvgF1)) +   
  geom_line(colour = "blue") +   
  geom_errorbar(
    aes(ymin = AvgF1 - ErrorF1, ymax = AvgF1 + ErrorF1), 
    width = 0.2, 
    colour = "blue"
  ) +   
  labs(
    title = "            Recursive Feature Elimination Avg F1-score 
                         Correlated with Feature Count",
    x = "Number of Features Selected",
    y = "Avg F1-score"
  ) +   
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"), # Center the title
    plot.margin = margin(t = 10, r = 50, b = 10, l = 20) # Adjust margins if needed
  )+
  scale_x_continuous(breaks = x_breaks) +   
  scale_y_continuous(breaks = seq(0.7, 1, by = 0.1), limits = c(0.7, 1)) +   
  theme_minimal()

p_zoomed <- ggplot(data, aes(x = NumFeatures, y = AvgF1)) +   
  geom_line(colour = "blue") +   
  geom_errorbar(
    aes(ymin = AvgF1 - ErrorF1, ymax = AvgF1 + ErrorF1), 
    width = 0.2, 
    colour = "blue"
  ) +   
  labs(
    title = "Recursive Feature Elimination Avg F1-score\nCorrelated with Feature Count",
    x = "Number of Features Selected",
    y = "Avg F1-score"
  ) +   
  scale_x_continuous(limits = c(1, 30), breaks = seq(1, 30, by = 5)) +   
  scale_y_continuous(breaks = seq(0.7, 1, by = 0.1), limits = c(0.7, 1)) +   
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"), # Center the title
    plot.margin = margin(t = 10, r = 50, b = 10, l = 20) # Adjust margins if needed
  )

# Display the plot
print(p)

print(p_zoomed)

# print the list of features with max f1 score
max_f1_row <- json_data[which.max(json_data$AvgF1), ]
print(max_f1_row)

# saving image
save_plot_as_jpeg <- function(plot, filename, width = 800, height = 600, units = "px", res = 150) {
  # Open a JPEG graphics device
  jpeg(filename, width = width, height = height, units = units, res = res)
  
  # Print the plot
  print(plot)
  
  # Close the graphics device
  dev.off()
}


save_plot_as_jpeg(p, "MNIST_plot")

