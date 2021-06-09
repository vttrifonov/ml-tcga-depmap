playground <- import('playground11')$'_playground11'

self<-reactive({
  .model<-req(input$model)
  playground(.model, NULL)
})

crispr_model_score <- reactive({
  .self<-req(self())
  .self$crispr_model_score %>%
    data.table(keep.rownames=TRUE) %>%
    setnames('True ', 'True', skip_absent=TRUE) %>%
    setnames(c('rn', 'True', 'False'), c('cols', 'train', 'test'), skip_absent=TRUE) %>%
    {.[, train:=train^2]} %>%
    { if ('test' %in% names(.)) .[, test:=test^2] else . } %>%
    {.[, train.ratio:=train/(n/m)]} %>%
    {.[order(-train.ratio)]} 
})

crispr_model_score_plot<-reactive({
  .crispr_model_score <- crispr_model_score()
  
  
  if ('test' %in% names(.crispr_model_score)) {
    .idx <- input$crispr_model_score_table_rows_selected
    p<-.crispr_model_score %>% copy %>%
      {
        ggplot(., aes(train, test))+
          geom_point(aes(text=cols), alpha=0.1)+
          geom_density2d()
      } 
    if (!is.null(.idx)) {
      p <- .crispr_model_score[.idx] %>%
        {
            p +
            geom_point(data=., color='red')+
            geom_text_repel(data=., aes(label=cols), color='red')
        }
    }
    p
  }
})

crispr_prediction <- reactive({
  .self<-req(self())
  .self$crispr_prediction
})

crispr_prediction_plot_data<-reactive({
  .crispr <- crispr_prediction()
  .crispr_model_score<-crispr_model_score()
  .idx <- req(input$crispr_model_score_table_rows_selected)
  
  cols<-.crispr_model_score$cols[.idx]
  plot_data <- suppressWarnings(.crispr$sel(cols=cols)$squeeze()$to_dataframe()) %>% data.table
})

crispr_prediction_depmap_plot <- reactive({
  plot_data <- crispr_prediction_plot_data() %>% copy
  plot_data <- plot_data[dataset=='DepMap'] %>%
    {.[, observed:=ifelse(observed, 'obs', 'pred')]} %>%
    dcast(row_label+source+train~observed, value.var='data')
  
  .add_title<-function(p, title) {
    add_annotations(
      p,
      text = title,
      x = 0.5,
      y = 1,
      yref = "paper",
      xref = "paper",
      xanchor = "middle",
      yanchor = "top",
      showarrow = FALSE,
      font = list(size = 15)
    )
  }
  .plot <-function(data, title, showlegend) {
    plot_ly(
      data,
      type='scatter', mode='markers', 
      x=~pred, y=~obs, color=~source,
      legendgroup=~source, showlegend=showlegend
    ) %>%
      .add_title(paste0(title, sprintf('(R2=%.0f%%)', 100*cor(data$pred, data$obs)^2))) %>%
      layout(
        xaxis=list(hoverformat='.2f'),
        yaxis=list(hoverformat='.2f')
      )
  }
  
  subplot(
    plot_data[train==FALSE] %>% .plot('test', FALSE),
    plot_data[train==TRUE] %>% .plot('train', TRUE),
    nrows=1, shareY = TRUE, titleX=TRUE
  )
})

crispr_prediction_plot <- reactive({
  plot_data <- crispr_prediction_plot_data() %>% copy
  plot_data[, color:=paste0(
    dataset, ',',
    ifelse(observed, 'obs', 'pred'), ',',
    ifelse(train, 'train', 'test')
  )]
  plot_data %>%
    {.[order(source)]} %>%
    plot_ly(
      type='scatter', mode='markers',
      x=~source, y=~data, color=~color
    ) %>%
    layout(
      yaxis=list(hoverformat='.2f')
    )
})
  




