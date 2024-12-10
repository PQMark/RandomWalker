#
# This is the server logic of a Shiny web application. You can run the
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

library(shiny)
library(shinyjs)

# Define server logic required to draw a histogram
function(input, output, session) {
  
  Data <- reactive({
    req(input$file)
    
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
  
  limit <- function(df, max_r = 10, head_r = 5, tail_r = 5, 
                    max_c = 10, head_c = 5, tail_c = 5) {
    # Limit rows
    if (nrow(df) > max_r) {
      df_rows <- rbind(
        head(df, head_r), 
        setNames(as.list(rep("-----", ncol(df))), names(df)),
        tail(df, tail_r)
      )
    } else {
      df_rows <- df
    }
    
    # Limit columns
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
  
  output$contents <- renderTable({
    df <- Data()
    
    limited_df <- limit(df)
    
    return(limited_df)
  }, sanitize.text.function = function(x) x)  # Allow separator text
  
  output$selected_model <- renderText({
    req(input$model)
    paste("You have selected:", input$model)
  })
  
  output$model_params <- renderUI({
    if (input$data_source == "upload") {
    req(input$file)}       # ensure the file is uploaded
   
    div(
      style = "padding-left: 15px; border-left: 2px solid #ccc; margin-top: 10px;", 
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
                 ), 
              
               )
             }, 
             "Permutation" = {
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
    
  
  
  output$advanced_params <- renderUI({
    #req(input$file)
    req(input$model %in% c("Boruta", "Permutation", "Recursive Feature Elimination"))
    
    tagList(
      useShinyjs(),
      
      tags$button(
        type = "button", 
        class = "btn btn-info", 
        `data-toggle` = "collapse",
        `data-target` = "#advancedParams",
        "Advanced Parameters"
      ),
    
      div(
        id = "advancedParams", 
        class = "collapse", 
        style = "margin-top: 10px;",
        
        div(
          style = "margin-bottom: 10px;", 
          checkboxInput(
            inputId = "default_optimization",
            label = "Default Optimization",
            value = FALSE
          )
        ),
        
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
        
        if (input$model == "Recursive Feature Elimination") {
         tagList(
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
    
  
  session$onFlushed(function() {
    session$sendCustomMessage(type = 'tooltip-init', message = NULL)
  })
  
  
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
  
  observeEvent(input$goButton, {
    
    # basic parameters
    numIteration <- input$numIteration
    numFolds <- input$numFolds
    
    if (input$model == "Recursive Feature Elimination") {
      numFeatures <- input$numFeatures
      
      InitialThreshold <- if (!is.null(input$InitialThreshold) && input$InitialThreshold != "") {
        as.numeric(input$InitialThreshold)
      } else {
        0.2  # default
      }
      
      decayFactor <- if (!is.null(input$decayFactor) && input$decayFactor != "") {
        as.numeric(input$decayFactor)
      } else {
        1.5  # default
      }
      
      lrParams <- list(
        InitialThreshold = InitialThreshold,
        decayFactor = decayFactor
      )
      
      # Optimization
      
    }
    
    
  })
  
}