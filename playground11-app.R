library(data.table)
library(magrittr)
library(reticulate)
library(ggplot2)
library(shiny)
library(plotly)
library(ggrepel)
source('common/module.R')
load.module('common')

if (FALSE) {
  input<-reactiveValues()
  input$crispr_model_score_table_rows_selected<-1
  input$model<-'20210608-0.8'
}

shinyApp(
  ui=fluidPage(
    selectizeInput(
      'model', 
      'Select model', 
      choices=list.files('.cache/playground11'),
      selected='full'
    ),
    fluidRow(
      column(4, 
        tags$label('Select gene', class='control-label'),
        DT::dataTableOutput('crispr_model_score_table')
      ),
      column(4, plotOutput('crispr_model_score_plot'))
    ),
    plotlyOutput('crispr_prediction_depmap_plot', height='400px'),
    plotlyOutput('crispr_prediction_plot', height='600px')
  ),
  
  server=function(input, output) {
    source('playground11-shiny.R', local=TRUE)
    
    output$crispr_model_score_table<-DT::renderDataTable({
      table<-crispr_model_score() %>% copy
      table[, train:=round(train, 2)]
      table[, train.ratio:=round(train.ratio, 2)]
      if ('test' %in% names(table)) 
        table[, test:=round(test, 2)]
    
      table %>%
        datatable_scroller(selection='single')
    }, server=TRUE)
    
    output$crispr_model_score_plot<-renderPlot(crispr_model_score_plot())
    
    output$crispr_prediction_depmap_plot<-renderPlotly(
      crispr_prediction_depmap_plot() %>%
        plotly_size(height='100%')
    )
    
    output$crispr_prediction_plot<-renderPlotly(
      crispr_prediction_plot() %>%
        plotly_size(height='100%')
    )
  }
)