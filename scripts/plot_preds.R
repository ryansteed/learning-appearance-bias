library(ggplot2)

output_path = "git/caliskan-image-bias/caliskan-retraining-inception/output"
traits = c("Attractive", "Competent", "Dominant", "Extroverted", "Likeable", "Trustworthy", "Threat")
coefs = c()

w = 8

for (trait in traits) {
  ### LOAD DATA ###
  preds = read.csv(sprintf("%s/preds_%s.csv", output_path, trait))
  preds_random = read.csv(sprintf("%s/preds_%s_random.csv", output_path, trait))
  
  ### SCATTERS ###
  scatter_source = ggplot(preds, aes(x=actual, y=pred, color=Source, stroke=0)) + 
    geom_point() +
    geom_smooth(method=lm, se=FALSE) +
    scale_colour_brewer("Dataset", palette="Set1") +
    labs(y="Predicted", x="Actual") +
    ggtitle(sprintf("Predicted %s Scores", trait)) +
    theme(legend.position=c(.85, .10))
  ggsave(sprintf("%s/plots/scatter-source_%s.png", output_path, trait), width=w)
  
  scatter_folds = ggplot(preds, aes(x=actual, y=pred, color=fold, stroke=0)) + 
    geom_point() +
    scale_colour_gradient2("Fold") +
    labs(y="Predicted", x="Actual") +
    ggtitle(sprintf("Predicted %s Scores", trait))
  ggsave(sprintf("%s/plots/scatter-folds_%s.png", output_path, trait), width=w)
  
  histo_random = ggplot(preds_random, aes(x=actual)) +
    geom_histogram()
  print(histo_random)
  
  scatter_random = ggplot(preds_random, aes(x=actual, y=pred, stroke=0)) + 
    geom_point(color="#e41a1c") +
    geom_smooth(method=lm, se=FALSE, color="#e41a1c") +
    labs(y="Predicted", x="Actual") +
    ggtitle(sprintf("Predicted %s Scores for 300 Random Faces", trait))
  ggsave(sprintf("%s/plots/scatter-random_%s.png", output_path, trait), width=w)
  
  ### CORRELATIONS ###
  library("Metrics")
  p = cor.test(preds$actual, preds$pred, method="pearson")
  p_random = cor.test(preds_random$actual, preds_random$pred, method="pearson")
  
  cors = c(
    p$estimate,
    p$p.value,
    rmse(preds$actual, preds$pred),
    p_random$estimate, 
    p_random$p.value,
    rmse(preds_random$actual, preds_random$pred)
  ) 
  coefs[[trait]] = cors
}

df = data.frame(coefs)
rownames(df) = c("Rho", "p-value", "RMSE", "Rho_random", "p-value_random", "RMSE_random")
df