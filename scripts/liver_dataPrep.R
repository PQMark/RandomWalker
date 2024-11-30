# Load X data
x_data <-readxl::read_excel('xdata (1).xlsx')

# Load Y data
y_data <- read.csv('ydata.csv', header = T)


# Match sample index across data frames. 
index <- match(x_data$Sample,paste('S', y_data$Animal.number, sep = ''))

# Ensure both dataframes are in same order and samples are matched. 
y_data <- y_data[index,] # Extract only matching samples (in correct order) from y.
x_data <- x_data[order(y_data$Animal.number),] # Reorder x to to that of ordered y.
x_data <- x_data[,-c(1,2)] # Remove superfluous x columns.
y_data <- y_data[order(y_data$Animal.number),] # Reorder y to that of ordered y.

# Extract the relevent variable from y_data.
y <-  y_data$Relative.liver.weight

data <- cbind(y,x_data)
write.csv(data, "data_liver_weight.csv", row.names = FALSE)

