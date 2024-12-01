# Load the jsonlite package
library(jsonlite)

# Define the path to your JSON file
json_file_path <- "results.json"

# Read the JSON file
json_data <- fromJSON(json_file_path)

#Sort
json_data <- json_data[order(json_data$Score), ]

#Rank
json_data$Features_Rank <- seq_len(nrow(json_data))


data <- data.frame(Rank = json_data$Features_Rank,  value = json_data$AvgPermutScore, sd = json_data$ErrorPermutScore)
data <- data[data$Rank >= 1 & data$Rank <= 20, ]

library(ggplot2)


p <- ggplot(data, aes(x = Feature, y = PermutationScore)) +
p <- ggplot(data) +
    geom_bar( aes(x=Feature, y=value), stat="identity", fill="#316734") +
    geom_errorbar( aes(x=name, ymin=value-sd, ymax=value+sd), width=0.4, colour="black", size=1.3)+
    labs(
    title = "Permutation based feature selection result",
    x = "Features Permuted",
    y = "Permutation Score"
  ) +   
  scale_x_continuous(breaks = seq(1, 21, by = 4)) +   
  scale_y_continuous(breaks = seq(0.7, 1, by = 0.1), limits = c(0.7, 1)) +   
  theme_minimal()

# Display the plot
print(p)
