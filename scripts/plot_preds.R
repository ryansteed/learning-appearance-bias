library(ggplot2)
library(dplyr)

output_path = "git/caliskan-image-bias/caliskan-retraining-inception/output/preds"
data_path = "git/caliskan-image-bias/data"
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


##### 23 Mar 2020 Experiments ####

## Nationality Bias Experiment ##
average_faces = read.csv(sprintf("%s/average-faces-preds_all.csv", output_path))
average_preds = average_faces %>%
  group_by(Source) %>%
  select(starts_with('pred_')) %>%
  summarise_all(mean) %>%
  tidyr::gather("trait", "pred", -Source)
average_preds

ggplot(average_preds, aes(x=Source, y=pred)) +
  geom_bar(stat="identity") +
  facet_wrap(~trait, ncol=2)

## Race Bias Experiment ##
race_faces = read.csv(sprintf("%s/people_all_aligned-preds_all.csv", output_path))
average_preds = race_faces %>%
  group_by(Source) %>%
  select(starts_with('pred_')) %>%
  summarise_all(mean) %>%
  tidyr::gather("trait", "pred", -Source) %>%
  tidyr::separate("Source", c("race", "gender"), sep=5)
average_preds

ggplot(average_preds, aes(x=race, y=pred, fill=gender)) +
  geom_bar(stat="identity") +
  facet_wrap(~trait, ncol=3)

# t-tests
gathered = race_faces %>%
  select(starts_with("pred"), "Source") %>%
  tidyr::gather("trait", "pred", -Source) %>%
  tidyr::separate("Source", c("race", "gender"), sep=5)
for (trait_val in unique(average_preds$trait)) {
  print(trait_val)
  
  trait_preds = gathered %>% filter(trait==trait_val)
  
  white = trait_preds %>% filter(race=="white")
  black = trait_preds %>% filter(race=="black")
  print(t.test(white$pred, black$pred))
  
  man = trait_preds %>% filter(gender=="man")
  woman = trait_preds %>% filter(gender=="woman")
  print(t.test(man$pred, woman$pred))
  
  # intersectional bias
  white_man = trait_preds %>% filter(race=="white", gender=="man")
  black_woman = trait_preds %>% filter(race=="black", gender=="woman")
  print(t.test(white_man$pred, black_woman$pred))
}


## Occupation Bias Experiment ##
occupations = read.csv(sprintf("%s/occupations_aligned-preds_all.csv", output_path))
competence_scores = read.table(sprintf("%s/occupations/competence_scores.txt", data_path), sep='\t', header=FALSE, col.names=c("Source", "1", "2"))
levels(competence_scores$Source) = tolower(levels(competence_scores$Source))
means = occupations %>%
  group_by(Source) %>%
  summarise(mean_competence_pred = mean(pred_Competent)) %>%
  arrange(-mean_competence_pred)
means

mean(means$mean_competence_pred)
levels(means$Source) = tolower(levels(means$Source))
occupation_competence = inner_join(means, competence_scores, by="Source")
scale = function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# correlation wtih Caliskan's comptence scores
occupation_competence$mean_competence_pred = scale(occupation_competence$mean_competence_pred)
occupation_competence$diffs = occupation_competence$X1 - occupation_competence$mean_competence_pred
plot(occupation_competence$mean_competence_pred, occupation_competence$X1)
cor.test(occupation_competence$mean_competence_pred, occupation_competence$X1, method="pearson")
write.csv(occupation_competence, sprintf("%s/preds-occupations_competence-scores.csv", output_path))


## Election Outcomes Experiment ##
politicians = read.csv(sprintf("%s/politicians-database_aligned-preds_all.csv", output_path))
politicians %>%
  group_by(Source) %>%
  summarise(mean(pred_Competent))

election_results = read.csv(file=sprintf("%s/politicians-database/coding.csv", data_path))
face_by_result = merge(politicians[c("Face.name", "pred_Competent", "pred_Trustworthy", "pred_Likeable", "pred_Attractive", "pred_Dominant", "pred_Extroverted")], election_results, by.x="Face.name", by.y="Full.Label")

## cor between predicted competence and ground truth competence?
cor.test(~ pred_Competent + Competency, face_by_result, method="pearson")

## by vote share
lm = lm(
  Vote.Share ~ 
    pred_Dominant #+ pred_Attractive + pred_Likeable + pred_Dominant + pred_Trustworthy + pred_Extroverted,
  + Race + Year + Gender + Race.Ethnicity + Incumbent. + Age,
  face_by_result
)
summary(lm)
plot(lm$residuals)

## by vote spread
# need to get only one of the candidates (random), then calculate votes for this candidate minus votes for the other over total - use that as target
pick_one = face_by_result[sample(nrow(face_by_result)),] %>%
  group_by(Election.ID) %>%
  filter(n() == 2) %>%
  sample_n(1)
joined = inner_join(pick_one, face_by_result[,c("Election.ID", "Votes", "Individual.ID")], by="Election.ID")
joined = joined[which(joined$Individual.ID.x != joined$Individual.ID.y),]
joined$Votes.x = as.numeric(gsub(",", "", sapply(strsplit(as.character(joined$Votes.x)," "), "[", 1)))
joined$Votes.y = as.numeric(gsub(",", "", sapply(strsplit(as.character(joined$Votes.y)," "), "[", 1)))
joined$vote_diff = joined$Votes.x - joined$Votes.y
joined$spread = joined$vote_diff/(joined$Votes.x + joined$Votes.y)
joined$pred_Trustworthy = scale(joined$pred_Trustworthy)
write.csv(joined, sprintf("%s/preds-politicians_election-results.csv", output_path))

plot(joined$pred_Competent, joined$spread)

lm_comp = lm(spread ~ pred_Competent, joined)
summary(lm_comp)
