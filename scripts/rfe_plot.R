# Load the jsonlite package
library(jsonlite)
library(ggplot2)
library(tools)
library(dplyr)

# Define the path to JSON file
# json_file_path <- "src/RandomWalker/temp/results.json"


# args <- commandArgs(trailingOnly = TRUE)
# if (length(args) == 0) {
#   stop("No file path provided. Please provide the path to the JSON file as an argument.")
# }

print(getwd())
json_file_name <- "RFE_FeatureImportance.json"    #args[1]
file_directory <- "src/RandomWalker/temp"
json_file_path <- file.path(file_directory, json_file_name)

print(json_file_path)

base_name <- file_path_sans_ext(basename(json_file_path))
output_plot_name <- paste0(base_name, "_plot.png")

# Read the JSON file
json_data <- fromJSON(json_file_path, simplifyVector = FALSE)

combined_df <- data.frame()

# Iterate through each sublist to append data with a group identifier
for (i in seq_along(json_data)) {
  
  # Extract the current sublist
  sublist <- json_data[[i]]
  
  # Convert the sublist to a data frame
  df <- data.frame(
    Group = paste("Group", i),
    Features = sapply(sublist, function(x) paste(x$Features, collapse = ",")),
    AvgF1 = sapply(sublist, function(x) x$AvgF1),
    ErrorF1 = sapply(sublist, function(x) x$ErrorF1),
    stringsAsFactors = FALSE
  )
  
  # Calculate the number of features
  df$NumFeatures <- sapply(strsplit(df$Features, ","), length)
  
  # Filter the data for NumFeatures between 1 and 20
  df <- df %>% filter(NumFeatures >= 0 & NumFeatures <= 70)
  
  print(df)

  # Append to the combined data frame
  combined_df <- bind_rows(combined_df, df)
}

# Check if combined_df is not empty
if (nrow(combined_df) == 0) {
  stop("No data available after filtering. Please check your JSON data and filtering criteria.")
}

# Create the combined plot using ggplot2 with facets arranged vertically
p <- ggplot(combined_df, aes(x = NumFeatures, y = AvgF1)) +   
  geom_line(colour = "blue") +   
  geom_errorbar(
    aes(ymin = AvgF1 - ErrorF1, ymax = AvgF1 + ErrorF1), 
    width = 0.2, 
    colour = "blue"
  ) +   
  labs(
    title = "Recursive Feature Elimination",
    x = "Number of Features Selected",
    y = "Avg F1-score"
  ) +   
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"), # Center the title
    plot.margin = margin(t = 10, r = 50, b = 10, l = 20) # Adjust margins if needed
  )+
  scale_x_continuous(breaks = x_breaks) +   
  scale_y_continuous(breaks = seq(0.7, 1, by = 0.1), limits = c(0.7, 1)) +   
  theme_minimal() +
  facet_wrap(~ Group, ncol = 1)  # Arrange facets vertically

# Adjust theme for better readability (optional)
p <- p + theme(
  strip.text = element_text(size = 12, face = "bold"),
  plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
)

print(p)
