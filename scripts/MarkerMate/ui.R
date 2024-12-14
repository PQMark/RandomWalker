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
              choices = c("Use MNIST Dataset" = "mnist"),
              selected = "upload"
            ),
          
            
            # File input -- Upload file
            
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
            uiOutput("advanced_params"), 
            
            # Run Button
            div(style = "margin-top: 20px;",
                actionButton(
                  inputId = "goButton", 
                  label = "Run", 
                  class = "btn btn-primary"
                )
            )
        ),
        
        # Main panel
        mainPanel(
          tableOutput("contents"), 
          br(), 
          h4("Selected Model:"), 
          textOutput("selected_model"), 
          conditionalPanel(
            condition = "input.data_source == 'mnist'",
            imageOutput("feature_plot")
          ), 
          tags$div(
            style = "margin-top: 20px;",
            tags$p("This plot shows the feature importance for the selected model. ",
                   "It helps visualize which features contribute most to the classification.")
          )
        )
      ), 
  
    tags$script(HTML("
    Shiny.addCustomMessageHandler('tooltip-init', function(message) {
        $('[data-toggle=\"tooltip\"]').tooltip();
    });
  "))
)
