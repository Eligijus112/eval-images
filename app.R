library(shiny)
library(data.table)
library(magrittr)
library(icesTAF)
library(reticulate)
py_install(c('scipy', 'numpy', 'pandas', 'tensorflow', 'opencv-python'))

import('numpy')
import('pandas')
import('cv2')
import('re')
import('os')

ui <- fluidPage(
   
   titlePanel("Image labelling application"),
   
   sidebarLayout(
      sidebarPanel(
        
        fileInput("img", "Choose an image",
                  multiple = FALSE, 
                  accept = c('image/png', 'image/jpeg')),
        
        actionButton('source_py',
                     "Evaluate image"), 
        
        actionButton('show_info', 
                     'Show evaluated information')
      ),
      
      mainPanel(
        plotOutput("barplot")
      )
   )
)

server <- function(input, output) {
   
  observeEvent(input$img, {
    mkdir('input')
    do.call(file.remove, list(list.files("input/", full.names = TRUE)))
    inFile <- input$img
    if (is.null(inFile)){
      return()
    }
    file.copy(inFile$datapath, file.path("input/", inFile$name))
  })
  
  observeEvent(input$source_py, {
    mkdir('output')
    withProgress(message = 'Evaluating images', value = 0, {
      system('python master.py')
      incProgress(1)
    })
  })
  
  observeEvent(input$show_info, {
    
    # browser()
    d <- fread('output/fitted_clases.csv') %>% 
      .[order(p)] 
    
    output$barplot <- renderPlot({
      par(las=2) 
      par(mar=c(5,8,4,2))
      barplot(d$p, names.arg = d$class_label, horiz = TRUE,  cex.names=0.9, 
              main = 'Probabilty of class', xpd = FALSE, col = adjustcolor('blue', 0.5))
    })
  })
}

shinyApp(ui = ui, server = server)

