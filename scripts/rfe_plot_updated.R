# Load the jsonlite package
library(jsonlite)
library(ggplot2)
library(tools)
library(dplyr)

# Define the path to JSON file
# json_file_path <- "src/RandomWalker/temp/RFE_FeatureImportance.json"


# args <- commandArgs(trailingOnly = TRUE)
# if (length(args) == 0) {
#   stop("No file path provided. Please provide the path to the JSON file as an argument.")
# }

combine_json_files <- function(json_file_paths, output_file) {
  combined_data <- list()  # Initialize an empty list for combined data
  
  for (i in seq_along(json_file_paths)) {
    # Load each JSON file
    file_path <- json_file_paths[i]
    json_data <- fromJSON(file_path)
    
    # Add a group identifier to the data
    combined_data[[paste("Group", i)]] <- json_data
  }
  
  # Write the combined data into a new JSON file
  toJSON(combined_data, pretty = TRUE, auto_unbox = TRUE) %>% 
    write(output_file)
  
  message(paste("Combined JSON file saved to:", output_file))
}

json_file_paths <- c("temp/results_rfe_fold_0.json", "temp/results_rfe_fold_1.json",
                    "temp/results_rfe_fold_2.json", "temp/results_rfe_fold_3.json",
                    "temp/results_rfe_fold_4.json"
                 )  # Replace with your file paths
output_file <- "temp/combined.json"  # Path for thae combined output file
combine_json_files(json_file_paths, output_file)

print(getwd())
json_file_name <- "combined.json"    #args[1]
file_directory <- "temp"
json_file_path <- file.path(file_directory, json_file_name)

print(json_file_path)

base_name <- file_path_sans_ext(basename(json_file_path))
output_plot_name <- paste0(base_name, "_plot.png")

# Read the JSON file
json_data <- fromJSON(json_file_path, simplifyVector = FALSE)

# Initialize the combined data frame
combined_df <- data.frame()

# Iterate through each sublist to append data with a group identifier
for (i in seq_along(json_data)) {
  # Extract the current sublist
  sublist <- json_data[[i]]
  
  # Ensure sublist is a list
  if (!is.list(sublist)) next
  
  # Convert the sublist to a data frame
  df <- data.frame(
    Group = paste("Group", i),
    Features = sapply(sublist, function(x) {
      if (is.list(x) && !is.null(x$Features)) {
        paste(x$Features, collapse = ",")
      } else {
        NA
      }
    }),
    AvgF1 = sapply(sublist, function(x) {
      if (is.list(x) && !is.null(x$AvgF1)) {
        x$AvgF1
      } else {
        NA
      }
    }),
    ErrorF1 = sapply(sublist, function(x) {
      if (is.list(x) && !is.null(x$ErrorF1)) {
        x$ErrorF1
      } else {
        NA
      }
    }),
    stringsAsFactors = FALSE
  )
  
  # Calculate the number of features
  df$NumFeatures <- sapply(strsplit(df$Features, ","), length)
  
  # Filter the data for NumFeatures between 1 and 70
  df <- df %>% filter(NumFeatures >= 1 & NumFeatures <= 70)
  
  # Print the data frame for debugging
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
    y = "Mean Test Accuracy"
  ) +   
  scale_x_continuous(breaks = seq(1, 21, by = 4)) +   
  scale_y_continuous(breaks = seq(0.7, 1, by = 0.1), limits = c(0.7, 1)) +   
  theme_minimal() +
  facet_wrap(~ Group, ncol = 1)  # Arrange facets vertically

# Adjust theme for better readability (optional)
p <- p + theme(
  strip.text = element_text(size = 12, face = "bold"),
  plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
)

p <- p + theme(
  panel.background = element_rect(fill = "white", color = "white"),
  plot.background = element_rect(fill = "white", color = "white")
)

# Save the plot with ggsave
ggsave("full_plot.png", plot = p, width = 10, height = 15, dpi = 300, bg = "white")
getwd()
print(p)

