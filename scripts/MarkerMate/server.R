#
# This is the server logic of a Shiny web application. You can run the
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

library(shiny)

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
    req(input$file)       # ensure the file is uploaded
    
    switch(input$model, 
           "Boruta" = {
             tagList(
               numericInput(
                 inputId = "numIteration", 
                 label = "Number of Iterations", 
                 value = 30, 
                 min = 1,
                 max = 50
               ), 
               
               numericInput(
                 inputId = "numFolds", 
                 label = "Number of Folds for CV", 
                 value = 5, 
                 min = 2
               ), 
               
               numericInput(
                 inputId = "Ntrees", 
                 label = "Number of Trees", 
                 value = 1000, 
                 min = 10, 
                 step = 100
               )
             )
           }, 
           "Permutation" = {
             
           }, 
           "Recursive Feature Elimination" = {
             
           },
           "mRMR" = {
             
           }
           )
  })
  
  output$advanced_params <- renderUI({
    #req(input$file)
    req(input$model %in% c("Boruta", "Permutation", "Recursive Feature Elimination"))
    
    tagList(
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
          numericInput(
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
            value = NULL
          )
        ), 
        
        div(
          style = "margin-bottom: 10px;", 
          numericInput(
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
            value = NULL
          )
        )
        
      )
      
    )
    
  })
  
  session$onFlushed(function() {
    session$sendCustomMessage(type = 'tooltip-init', message = NULL)
  })
  
}