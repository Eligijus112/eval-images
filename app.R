library(shiny)
library(data.table)
library(magrittr)
library(icesTAF)
library(reticulate)
library(jpeg)
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
                  multiple = TRUE, 
                  accept = c('image/png', 'image/jpeg')),
        
        actionButton('source_py',
                     "Evaluate images"), 
        
        actionButton('show', 'Show evaluation'),
        
        selectInput('Image', 'Select image', "")
      ),
      
      mainPanel(
        plotOutput("barplot")
      )
   )
)

server <- function(input, output, session) {
   
  observeEvent(input$img, {
    unlink('input', recursive = TRUE)
    unlink('output', recursive = TRUE)
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
  
  observeEvent(input$show, {
    
    if(length(grep('fitted_clases.csv', list.files('output/')))!=0 & !is.null(input$Image)){
      d <- fread('output/fitted_clases.csv') %>% 
        .[order(p)] %>%
        .[path == paste0("input/", input$Image)]
      
      output$barplot <- renderPlot({
        par(las=2) 
        par(mar=c(5,8,4,2))
        barplot(d$p, names.arg = d$class_label, horiz = TRUE,  cex.names=0.9, 
                main = 'Probabilty of class', xpd = FALSE, col = adjustcolor('blue', 0.5))
      })
    }
  })
  
  observeEvent(input$Image, {
    
    if(length(grep('fitted_clases.csv', list.files('output/')))!=0){
      d <- fread('output/fitted_clases.csv') %>% 
        .[order(p)] %>%
        .[path == paste0("input/", input$Image)]
      
        output$barplot <- renderPlot({
          par(las=2) 
          par(mar=c(5,8,4,2))
          barplot(d$p, names.arg = d$class_label, horiz = TRUE,  cex.names=0.9, 
                  main = 'Probabilty of class', xpd = FALSE, col = adjustcolor('blue', 0.5))
        })
      }
    })
  
  outVar = reactive({
    if(!is.null(input$img)){
      input$img$name
    }
  })
  
  observe({
    if(!is.null(outVar())){
      updateSelectInput(session, "Image",
                        choices = outVar()
      )}
    })
}

shinyApp(ui = ui, server = server)