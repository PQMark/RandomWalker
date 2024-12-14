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
library(jsonlite)

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
                 )
                 
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
            multiple = FALSE,
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
            multiple = FALSE,
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
            multiple = FALSE,
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
  
toggle_pic <- reactiveVal(FALSE)
  
  observeEvent(input$goButton, {
    
    toggle_pic(FALSE)
    
    # check the temp folder
    temp_dir <- normalizePath("../../temp", mustWork = FALSE)
    if (!dir.exists(temp_dir)) {
      dir.create(temp_dir, recursive = TRUE)
    }
    
    # Initialize a list to store all parameters
    overall_params <- list()
    
    # Dataset information
    overall_params$data_source <- input$data_source
    
    if (input$data_source == "mnist") {
        overall_params$dataset_params <- list(
          num_samples = input$num_samples,
          digits = as.numeric(input$digits)
        )
    } else {
      # overall_params$dataset_params$file_path <- input$file$datapath
      
      save_dir <- ""
      save_path <- file.path(save_dir, input$file$name)
      
      file.copy(input$file$datapath, save_path)
      overall_params$dataset_params$file_path <- input$file$name
      
    }
    
    # Selected method
    overall_params$method <- input$model
    
    # Shared param 
    overall_params$model_params$numIteration <- input$numIteration
    
    # Shared advanced params for three of the models
    if (input$model %in% c("Boruta", "Recursive Feature Elimination", "Permutation")) {
        
        # NTrees
        overall_params$advanced_params$Ntrees <- if (!is.null(input$Ntrees) && input$Ntrees != "") {
          as.numeric(input$Ntrees)
        } else {
          1000
        }
        
        # NumLeaf
        overall_params$advanced_params$num_leaves <- if (!is.null(input$num_leaves) && input$num_leaves != "") {
          as.numeric(input$num_leaves)
        } else {
          0
        }
        
        # Max-Depth
        overall_params$advanced_params$max_depth <- if (!is.null(input$max_depth) && input$max_depth != "") {
          as.numeric(input$max_depth)
        } else {
          0
        }
    }
    
    # Method-specific parameters
    if (input$model == "mRMR") {
      overall_params$model_params$binSize <- input$binSize
      overall_params$model_params$maxFeatures <- input$maxFeatures
    }
    
    if (input$model == "Recursive Feature Elimination") {
      overall_params$model_params$numFeatures <- input$numFeatures
      
      overall_params$advanced_params$InitialThreshold <- if (!is.null(input$InitialThreshold) && input$InitialThreshold != "") {
        as.numeric(input$InitialThreshold)
      } else {
        0.2  
      }
      
      overall_params$advanced_params$decayFactor <- if (!is.null(input$decayFactor) && input$decayFactor != "") {
        as.numeric(input$decayFactor)
      } else {
        1.5 
      }
    }
    
    # Boruta has no specific parameters
    
    
    # store the parameters
    json_overall_params <- toJSON(overall_params, auto_unbox = TRUE, pretty = TRUE)
    json_file_path <- file.path(temp_dir, "overall_params_R.json")
    write(json_overall_params, json_file_path)
    
    
    # run GO
    cur <- getwd()
    go_path <- normalizePath("../../", mustWork = FALSE)
    setwd(go_path)
    system2("go", args = c("run", "."), wait = TRUE)
    
    setwd(cur)

    # Construct image path based on the method
    method_name <- input$model
    if (method_name == "Recursive Feature Elimination") {
      method_name <- "RFE"
    }
    img_filename <- paste0(method_name, "_FeatureImportance_plot.png")
    img_path <- file.path("../../temp", img_filename)
    img_path <- normalizePath(img_path, mustWork = FALSE)
    
    print(paste("Image Path:", img_path))
    
    if (input$data_source == "mnist" && file.exists(img_path)) {
      toggle_pic(TRUE)
    } else {
      toggle_pic(FALSE)
      if (input$data_source == "mnist") {
        showNotification("Feature importance plot not found. Please ensure the Go script ran successfully.", type = "error")
      }
    }
    
    
  })
  
  output$feature_plot <- renderImage({
    req(toggle_pic())  # Only render if toggle_pic is TRUE
    
    # Construct the image path
    method_name <- input$model
    if (method_name == "Recursive Feature Elimination") {
      method_name <- "RFE"
    }
    img_path <- normalizePath(file.path("../../temp", paste0(method_name, "_FeatureImportance_plot.png")), mustWork = FALSE)
    
    if (!file.exists(img_path)) {
      warning(paste("Image path does not exist:", img_path))
      return(NULL)
    }
    
    print(paste("Rendering image from path:", img_path))
    
    list(
      src = img_path,
      contentType = "image/png",
      alt = "Feature Importance Plot", 
      width = "90%", 
      height = "auto"
    )
  }, deleteFile = FALSE)
  
}