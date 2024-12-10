#
# This is the server logic of a Shiny web application. 
# The server function handles reactive expressions, 
# sets up output rendering, and responds to user input.
#

library(shiny)
library(shinyjs)

# Define server logic required to draw a histogram and manage UI interactions
function(input, output, session) {
  
  # Reactive expression to read and store the uploaded data.
  # This will run whenever 'input$file' changes.
  Data <- reactive({
    req(input$file)  # Ensure a file has been uploaded before proceeding.
    
    # Use tryCatch to handle any errors when reading the CSV file.
    # If an error occurs, it returns a "safeError".
    tryCatch(
      {
        df <- read.csv(input$file$datapath, 
                       header = FALSE, 
                       stringsAsFactors = FALSE)
      }, 
      error = function(e) {
        stop(safeError(e))
      }
    )
    
    return(df)
  })
  
  # A helper function to limit rows and columns for display.
  # This is useful if the dataset is huge, so we only show 
  # a small "preview" with head and tail rows and partial columns.
  limit <- function(df, max_r = 10, head_r = 5, tail_r = 5, 
                    max_c = 10, head_c = 5, tail_c = 5) {
    # Limit rows first:
    # If more than max_r rows, show head_r rows, a separator row of "-----", and tail_r rows.
    if (nrow(df) > max_r) {
      df_rows <- rbind(
        head(df, head_r), 
        setNames(as.list(rep("-----", ncol(df))), names(df)),
        tail(df, tail_r)
      )
    } else {
      df_rows <- df
    }
    
    # Then limit columns:
    # If more than max_c columns, show first head_c columns, a "..." column, and last tail_c columns.
    if (ncol(df_rows) > max_c) {
      first_cols <- df_rows[, 1:head_c, drop = FALSE]
      dots <- data.frame("..." = rep("...", nrow(df_rows)), stringsAsFactors = FALSE)
      last_cols <- df_rows[, (ncol(df_rows)-tail_c+1):ncol(df_rows), drop = FALSE]
      df_cols <- cbind(first_cols, dots, last_cols)
    } else {
      df_cols <- df_rows
    }
    
    return(df_cols)
  }
  
  # Render the contents of the uploaded data as a table.
  # Uses the 'Data()' reactive and 'limit()' function for a formatted preview.
  output$contents <- renderTable({
    df <- Data()
    limited_df <- limit(df)
    return(limited_df)
  }, sanitize.text.function = function(x) x)  # Allow separator text ("-----" rows)
  
  # Render text that shows which model is selected.
  # 'req(input$model)' ensures that a model is selected before rendering.
  output$selected_model <- renderText({
    req(input$model)
    paste("You have selected:", input$model)
  })
  
  # Render a UI block of parameters that depend on the selected model and data source.
  # This UI is dynamically generated using 'renderUI' and a 'switch' statement on input$model.
  output$model_params <- renderUI({
    # If data source is "upload", we require a file be uploaded before showing params.
    if (input$data_source == "upload") {
      req(input$file)
    }
    
    # Add a styled container for the model parameters.
    div(
      style = "padding-left: 15px; border-left: 2px solid #ccc; margin-top: 10px;", 
      
      # Switch among different sets of parameters for each model:
      switch(input$model, 
             "Boruta" = {
               tagList(
                 numericInput(
                   inputId = "numIteration", 
                   label = span(style = "font-size: 12px;", "Number of Iterations"),
                   value = 30, 
                   min = 1,
                   max = 50
                 ), 
                 
                 numericInput(
                   inputId = "numFolds", 
                   label = span(style = "font-size: 12px;", "Number of Folds for CV"),
                   value = 5, 
                   min = 2
                 )
               )
             }, 
             "Permutation" = {
               tagList(
                 numericInput(
                   inputId = "numIteration", 
                   label = span(style = "font-size: 12px;", "Number of Iterations"),
                   # If data source is "mnist", default is 50, otherwise 30
                   value = if (input$data_source == "mnist") 50 else 30,  
                   min = 1,
                   max = 50
                 ), 
                 
                 numericInput(
                   inputId = "numFolds", 
                   label = span(style = "font-size: 12px;", "Number of Folds for CV"),
                   value = 5, 
                   min = 2
                 )
               )
             }, 
             "Recursive Feature Elimination" = {
               tagList(
                 numericInput(
                   inputId = "numIteration", 
                   label = span(style = "font-size: 12px;", "Number of Iterations"),
                   value = if (input$data_source == "mnist") 50 else 30,  
                   min = 1,
                   max = 50
                 ), 
                 
                 numericInput(
                   inputId = "numFolds", 
                   label = span(style = "font-size: 12px;", "Number of Folds for CV"),
                   value = 5, 
                   min = 2
                 ), 
                 
                 numericInput(
                   inputId = "numFeatures", 
                   label = span(style = "font-size: 12px;", "Min Number of Features"),
                   value = if (input$data_source == "mnist") 30 else 1,
                   min = 1, 
                   step = 10
                 )
               )
             },
             "mRMR" = {
               tagList(
                 numericInput(
                   inputId = "numIteration", 
                   label = span(style = "font-size: 12px;", "Number of Iterations"),
                   value = if (input$data_source == "mnist") 50 else 30,  
                   min = 1,
                   max = 50
                 ), 
                 
                 numericInput(
                   inputId = "numFolds", 
                   label = span(style = "font-size: 12px;", "Number of Folds for CV"),
                   value = 5, 
                   min = 2
                 ), 
                 
                 numericInput(
                   inputId = "binSize", 
                   label = span(style = "font-size: 12px;", "Number of Bins"),
                   value = 15,
                   min = 1, 
                   step = 1
                 ), 
                 
                 numericInput(
                   inputId = "maxFeatures", 
                   label = span(style = "font-size: 12px;", "Max Number of Selected Features"),
                   value = 20,
                   min = 1, 
                   step = 1
                 )
               )
             }
      )
    )
  })
    
  
  # Render a UI for advanced parameters.
  # These are collapsed by default and can be toggled by a button.
  output$advanced_params <- renderUI({
    # For advanced params, we require a model that uses these parameters.
    req(input$model %in% c("Boruta", "Permutation", "Recursive Feature Elimination"))
    
    tagList(
      useShinyjs(),
      
      # A button to toggle the collapse of advanced parameters
      tags$button(
        type = "button", 
        class = "btn btn-info", 
        `data-toggle` = "collapse",
        `data-target` = "#advancedParams",
        "Advanced Parameters"
      ),
    
      # Collapsible panel containing the advanced parameters
      div(
        id = "advancedParams", 
        class = "collapse", 
        style = "margin-top: 10px;",
        
        # Checkbox to toggle default optimization
        div(
          style = "margin-bottom: 10px;", 
          checkboxInput(
            inputId = "default_optimization",
            label = "Default Optimization",
            value = FALSE
          )
        ),
        
        # Selectize input for number of trees with a tooltip
        div(
          style = "margin-bottom: 10px;", 
          selectizeInput(
            inputId = "Ntrees", 
            label = tags$span(
              "Number of Trees:", 
              tags$span(
                class = "glyphicon glyphicon-question-sign", 
                style = "cursor: pointer; margin-left: 5px;", 
                `data-toggle` = "tooltip",
                title = "Default is 1000 if left blank."
              )
            ),
            choices = NULL,
            multiple = TRUE,
            options = list(create = TRUE, placeholder = "Default: 1000")
          )
        ),
        
        # Selectize input for number of leaves with a tooltip
        div(
          style = "margin-bottom: 10px;", 
          selectizeInput(
            inputId = "num_leaves", 
            label = tags$span(
              "Number of Leaves:", 
              tags$span(
                class = "glyphicon glyphicon-question-sign", 
                style = "cursor: pointer; margin-left: 5px;", 
                `data-toggle` = "tooltip",
                title = "Default is 10 if left blank."
              )
            ),
            choices = NULL,
            multiple = TRUE,
            options = list(create = TRUE, placeholder = "Default: 10")
          )
        ), 
        
        # Selectize input for max depth with a tooltip
        div(
          style = "margin-bottom: 10px;", 
          selectizeInput(
            inputId = "max_depth", 
            label = tags$span(
              "Max Depth:", 
              tags$span(
                class = "glyphicon glyphicon-question-sign", 
                style = "cursor: pointer; margin-left: 5px;", 
                `data-toggle` = "tooltip",
                title = "Default is 5 if left blank."
              )
            ),
            choices = NULL,
            multiple = TRUE,
            options = list(create = TRUE, placeholder = "Enter values (e.g. 20,30)")
          )
        ), 
        
        # Additional inputs appear if the model is "Recursive Feature Elimination"
        if (input$model == "Recursive Feature Elimination") {
         tagList(
          # Input for Initial Threshold (initial ratio of feature removal)
          div(
            style = "margin-bottom: 10px;", 
            selectizeInput(
              inputId = "InitialThreshold", 
              label = tags$span(
                "Initial Ratio:", 
                tags$span(
                  class = "glyphicon glyphicon-question-sign", 
                  style = "cursor: pointer; margin-left: 5px;", 
                  `data-toggle` = "tooltip",
                  title = "The initial percentage of features to drop. Default value is 0.2."
                )
              ),
              choices = NULL,
              multiple = FALSE,
              options = list(create = TRUE, placeholder = "Default: 0.2", maxItems = '1')
            )
          ),
          
          # Input for decay factor (how fast ratio decreases)
          div(
            style = "margin-bottom: 10px;", 
            selectizeInput(
              inputId = "decayFactor", 
              label = tags$span(
                "Decay Factor:", 
                tags$span(
                  class = "glyphicon glyphicon-question-sign", 
                  style = "cursor: pointer; margin-left: 5px;", 
                  `data-toggle` = "tooltip",
                  title = "Control how fast the removing ratio will decay. Default value is 1.5."
                )
              ),
              choices = NULL,
              multiple = FALSE,
              options = list(create = TRUE, placeholder = "Default: 1.5", maxItems = '1')
            )
          )
         )
          
        }
        
      )
      
    )
    
  })
    
  
  # After the UI is loaded, send a custom message to initialize tooltips using JavaScript.
  session$onFlushed(function() {
    session$sendCustomMessage(type = 'tooltip-init', message = NULL)
  })
  
  
  # Observe the "Default Optimization" checkbox:
  # If default_optimization is TRUE, disable certain input controls (Ntrees, max_depth, num_leaves).
  # Otherwise, enable them.
  observeEvent(input$default_optimization, {
    if (input$default_optimization) {
      shinyjs::disable("num_leaves") 
      shinyjs::disable("max_depth")
      shinyjs::disable("Ntrees")
    } else {
      shinyjs::enable("num_leaves")
      shinyjs::enable("max_depth")
      shinyjs::enable("Ntrees")
    }
  })
  
  # Observing the "goButton" (presumably a button to run the feature selection or 
  # model training process) and retrieving the selected parameters.
  observeEvent(input$goButton, {
    
    # Retrieve basic parameters from inputs
    numIteration <- input$numIteration
    numFolds <- input$numFolds
    
    # If the model is RFE, retrieve additional parameters
    if (input$model == "Recursive Feature Elimination") {
      numFeatures <- input$numFeatures
      
      # Extract InitialThreshold from the input, use default if empty
      InitialThreshold <- if (!is.null(input$InitialThreshold) && input$InitialThreshold != "") {
        as.numeric(input$InitialThreshold)
      } else {
        0.2  # default
      }
      
      # Extract decayFactor from the input, use default if empty
      decayFactor <- if (!is.null(input$decayFactor) && input$decayFactor != "") {
        as.numeric(input$decayFactor)
      } else {
        1.5  # default
      }
      
      lrParams <- list(
        InitialThreshold = InitialThreshold,
        decayFactor = decayFactor
      )
      
      # Here you could add code to run the RFE algorithm with the given parameters...
      # (Optimization steps not fully implemented in this snippet.)
    }
    
    # Similar logic could be applied for other models, using their respective parameters.
    
  })
  
}
