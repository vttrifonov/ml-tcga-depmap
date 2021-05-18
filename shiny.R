depmap_gdc_fit_model <- import('depmap_gdc_fit')$model

model <- reactive({
  depmap_gdc_fit_model(input$train_split, input$dims)
})

stats<-reactive({
  model()$stats %>%
    data.table %>%
    {.[, train:=round(train, 2)]} %>%
    {.[, test:=round(test, 2)]} %>%
    {.[, rand:=round(rand, 2)]} %>%
    {.[]} %>%
    datatable_scroller(options=list(dom='t', scrollY=300), selection='single')
})

plot_data<-reactive({
  .model <- model()
  .idx <- req(input$stats_rows_selected)
  .model$data(.idx-1)  
})

plot_stats1<-reactive({
  .model <- model()
  
  .model$stats %>% 
    data.table(keep.rownames=TRUE) %>% 
    { 
      ggplot(., aes(train, test))+
        geom_point()
    }
})

plot_stats2<-reactive({
  .model <- model()
  
  .model$stats %>% 
    data.table(keep.rownames=TRUE) %>% 
    { 
      ggplot(., aes(sort(train), sort(rand)))+
        geom_point()+
        geom_abline(slope=1, intercept=0)
    }
})

plot_data1<-reactive({
  .data<-plot_data()
  
  .data[[1]] %>%
    data.table %>%
    { 
      ggplot(., aes(pred, obs, text=CCLE_Name))+
        geom_point()+
        labs(title='train')
    } %>%
    ggplotly(tooltip=c('x', 'y', 'text'))
})

plot_data2<-reactive({
  .data<-plot_data()
  
  .data[[2]] %>%
    data.table %>%
    { 
      ggplot(., aes(pred, obs, text=CCLE_Name))+
        geom_point()+
        labs(title='train')
    } %>%
    ggplotly(tooltip=c('x', 'y', 'text'))
})

plot_data3<-reactive({
  .data<-plot_data()
  
  .data[[3]] %>%
    data.table(keep.rownames=TRUE) %>%
    { 
      ggplot(., aes(x=as.numeric(rn), y=expr, color=project_id, shape=is_normal))+
        geom_point()+
        labs(x='')
    } %>%
    ggplotly(tooltip=c('expr', 'project_id', 'is_normal'), height=500)
})


