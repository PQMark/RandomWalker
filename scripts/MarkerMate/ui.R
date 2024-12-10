#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

library(shiny)

# Define UI for application that draws a histogram
fluidPage(

    # Application title
    div(style = "text-align: center;",
        h1("Welcome to MarkerMate")
    ),

    sidebarLayout(
      
        sidebarPanel(
            # File input
            radioButtons(
              inputId = "data_source",
              label = "Data Source",
              choices = c("Upload CSV" = "upload", "Use MNIST Dataset" = "mnist"),
              selected = "upload"
            ),
            
            # File input -- Upload file
            conditionalPanel(
              condition = "input.data_source == 'upload'",
              fileInput(
                inputId = "file", 
                label = "Choose CSV File", 
                accept = c(".csv")
              )
            ),
            
            conditionalPanel(
              condition = "input.data_source == 'upload'",
              tags$div(
                style = "padding-left: 15px; border-left: 2px solid #ccc; margin-top: 10px;",
                
                selectInput(
                  inputId = "missing",
                  label = tags$span(style = "font-size: 12px;", "Missing value processing method"),
                  choices = c(
                    "Delete the sample",
                    "Take mean of the feature"
                  )
                )
              )
            ),
            
            # File input -- Use MNIST
            conditionalPanel(
              condition = "input.data_source == 'mnist'",
              tags$div(
                style = "padding-left: 15px; border-left: 2px solid #ccc; margin-top: 10px;",
                numericInput(
                  inputId = "num_samples",
                  label = tags$span(style = "font-size: 12px;", "Number of Samples"),
                  value = 100,
                  min = 1
                ),
                selectInput(
                  inputId = "digits",
                  label = tags$span(style = "font-size: 12px;", "Digits to Include"),
                  choices = 0:9,
                  selected = c(1, 2),
                  multiple = TRUE
                )
              )
            ),
            
            
            
        
            # Model selection
            selectInput(
              inputId = "model", 
              label = "Select Model", 
              choices = c(
                "Boruta", 
                "Permutation", 
                "Recursive Feature Elimination", 
                "mRMR"
              )
            ),
            
            # Model-specific HPs 
            uiOutput("model_params"),
            
            # Panel for Optional Hps
            uiOutput("advanced_params")
        ),
        
        # Main panel
        mainPanel(
          tableOutput("contents"), 
          br(), 
          h4("Selected Model:"), 
          textOutput("selected_model")
        )
      ), 
  
    tags$script(HTML("
    Shiny.addCustomMessageHandler('tooltip-init', function(message) {
        $('[data-toggle=\"tooltip\"]').tooltip();
    });
  "))
)
