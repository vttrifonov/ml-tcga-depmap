library(data.table)
library(magrittr)
library(reticulate)
library(ggplot2)
library(shiny)
library(plotly)
source('common/module.R')
load.module('common')

shinyApp(
  ui=fluidPage(
    fluidRow(
      column(2, numericInput('train_split', 'Train split', 0.8, min=0, max=1)),
      column(2, numericInput('dims', 'Dims', 400, min=1, max=800))
    ),
    fluidRow(
      column(4, DT::dataTableOutput('stats')),
      column(4, plotOutput('plot_stats1')),
      column(4, plotOutput('plot_stats2'))
    ),
    fluidRow(
      column(4, plotlyOutput('plot_data1')),
      column(4, plotlyOutput('plot_data2'))
    ),
    plotlyOutput('plot_data3')
  ),
  
  server=function(input, output) {
    source('shiny.R', local=TRUE)
    output$plot_stats1<-renderPlot(plot_stats1())
    output$plot_stats2<-renderPlot(plot_stats2())
    output$plot_data1<-renderPlotly(plot_data1())
    output$plot_data2<-renderPlotly(plot_data2())
    output$plot_data3<-renderPlotly(plot_data3())
    output$stats<-DT::renderDataTable(stats())
  }
)