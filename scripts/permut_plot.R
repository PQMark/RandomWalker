# Load the jsonlite package
library(jsonlite)
library(ggplot2)

# Define the path to your JSON file
json_file_path <- "src/RandomWalker/temp/METABRIC_RNA_Mutation_20_500_15_2.json"

# Read the JSON file
json_data <- fromJSON(json_file_path)

#Sort
#json_data <- json_data[order(json_data$Score), ]

#Rank
json_data$Features_Rank <- seq_len(nrow(json_data))


data <- data.frame(Rank = json_data$Features_Rank,  value = json_data$AvgF1, sd = json_data$ErrorF1)
data <- data[data$Rank >= 1 & data$Rank <= 20, ]

library(ggplot2)


p <- ggplot(data, aes(x = Rank, y = value)) +
    geom_bar(stat="identity", fill="#316734") +
    geom_errorbar(aes(ymin=value-sd, ymax=value+sd), width=0.4, colour="black", size=1.3)+
    labs(
    title = "Permutation based feature selection result",
    x = "Features Permuted",
    y = "Permutation Score"
  ) +   
  scale_x_continuous(breaks = seq(1, 21, by = 4)) +   
  scale_y_continuous(breaks = seq(0.7, 1, by = 0.1), limits = c(-1, 1)) +   
  theme_minimal()

# Display the plot
p
