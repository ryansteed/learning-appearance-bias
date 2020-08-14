library(ggplot2)
library(dplyr)
library(caret)
library(imager)

base_output_path = "git/learning-appearance-bias/output"
output_path = "git/learning-appearance-bias/output/preds"
data_path = "git/learning-appearance-bias/data"
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
  ggsave(sprintf("%s/plots/scatter-source_%s.png", base_output_path, trait), width=w)
  
  scatter_folds = ggplot(preds, aes(x=actual, y=pred, color=fold, stroke=0)) + 
    geom_point() +
    scale_colour_gradient2("Fold") +
    labs(y="Predicted", x="Actual") +
    ggtitle(sprintf("Predicted %s Scores", trait))
  ggsave(sprintf("%s/plots/scatter-folds_%s.png", base_output_path, trait), width=w)
  
  histo_random = ggplot(preds_random, aes(x=actual)) +
    geom_histogram() +
    ggtitle(sprintf("Ground Truth %s Scores", trait))
  print(histo_random)
  
  histo_random_pred = ggplot(preds_random, aes(x=pred)) +
    geom_histogram() +
    ggtitle(sprintf("Predicted %s Scores", trait))
  print(histo_random_pred)
  
  scatter_random = ggplot(preds_random, aes(x=actual, y=pred, stroke=0)) + 
    geom_point(color="#e41a1c") +
    geom_smooth(method=lm, se=FALSE, color="#e41a1c") +
    labs(y="Predicted", x="Actual") +
    ggtitle(sprintf("Predicted %s Scores for 300 Random Faces", trait))
  ggsave(sprintf("%s/plots/scatter-random_%s.png", base_output_path, trait), width=w)
  
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

## Random Face Error Analysis ##
for (trait in traits) {
  preds_trustworthy = read.csv(sprintf("%s/preds_Trustworthy_random.csv", output_path, trait)) %>%
    mutate(preds_bin = pred > 0) %>%
    mutate(actual_bin = actual > 0)
  cm = confusionMatrix(factor(preds_trustworthy$preds_bin), factor(preds_trustworthy$actual_bin))
  fourfoldplot(cm$table, color=c("red", "green"), main=sprintf("Confusion Matrix - %s", trait))
}


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
face_by_result = face_by_result %>%
  mutate(Competency = (Competency - mean(Competency, na.rm=TRUE)) / sd(Competency, na.rm=TRUE) * 100)
pick_one = face_by_result[sample(nrow(face_by_result)),] %>%
  group_by(Election.ID) %>%
  filter(n() == 2) %>%
  sample_n(1)
joined = inner_join(pick_one, face_by_result[,c("Face.name", "Election.ID", "Votes", "Individual.ID", "Competency", "pred_Competent")], by="Election.ID")
joined = joined[which(joined$Individual.ID.x != joined$Individual.ID.y),]
joined$Votes.x = as.numeric(gsub(",", "", sapply(strsplit(as.character(joined$Votes.x)," "), "[", 1)))
joined$Votes.y = as.numeric(gsub(",", "", sapply(strsplit(as.character(joined$Votes.y)," "), "[", 1)))
joined$vote_diff = joined$Votes.x - joined$Votes.y
joined$spread = joined$vote_diff/(joined$Votes.x + joined$Votes.y)
joined$spread_competency_ground = joined$Competency.x - joined$Competency.y
joined$spread_competency_pred = joined$pred_Competent.x - joined$pred_Competent.y
write.csv(joined, sprintf("%s/preds-politicians_election-results.csv", output_path))

### which elections have the biggest difference in ground truth competence?
sorted = joined %>%
  arrange(-spread_competency_ground)
write.csv(sorted[,c(
  "Face.name.x", "Face.name.y", "spread", "spread_competency_ground", "spread_competency_pred",
  "pred_Competent.x", "pred_Competent.y", "Competency.x", "Competency.y"
)], sprintf("%s/politicians-spread.csv", output_path))
### which elections have the biggest difference in predicted competence?
sorted_pred = joined %>%
  arrange(-spread_competency_pred)
sorted_pred[,c(
  "Face.name.x", "Face.name.y", "spread",
  "pred_Competent.x", "Competency.x", "pred_Competent.y", "Competency.y"
)]

### corr between predicted diff in scores and actual diff? no reason there should be
plot(joined$spread_competency_pred, joined$spread_competency_ground)
cor.test(~ spread_competency_pred + spread_competency_ground, joined, method="pearson")
### cor between predicted diff in scores and vote spread
plot(joined$spread_competency_pred, joined$spread)
cor.test(~ spread_competency_pred + spread, joined, method="pearson")
### cor between actual diff in scores and vote spread
plot(joined$spread_competency_ground, joined$spread)
cor.test(~ spread_competency_ground + spread, joined, method="pearson")

## error analysis
binarize = function(x) x > 0
preds_bin = face_by_result %>%
  select(starts_with("pred_"), "Face.name", "Competency", "Attractiveness", "Vote.Share") %>%
  rename(Attractiveness.Truth=Attractiveness) %>%
  mutate(Competency_Norm = (Competency - mean(Competency, na.rm=TRUE)) / sd(Competency, na.rm=TRUE) * 100) %>%
  mutate(error = abs(Competency_Norm - pred_Competent)) %>%
  mutate(Pred_Competent.Bin = pred_Competent > 0) %>%
  mutate(Competency.Bin = Competency_Norm > 0)
### ground-truth dist
ggplot(preds_bin, aes(x=Competency_Norm)) +
  geom_histogram()
ggplot(preds_bin, aes(x=pred_Competent)) +
  geom_histogram()
### confusion matrix
cm = confusionMatrix(factor(preds_bin$Pred_Competent.Bin), factor(preds_bin$Competency.Bin))
fourfoldplot(cm$table, color=c("red", "green"), main=sprintf("Confusion Matrix - Politician Competency", trait))
cm
### outlier errors?
ggplot(preds_bin, aes(x=error)) +
  geom_histogram()

sorted = preds_bin %>%
  arrange(desc(error))
for (imgname in sorted[1:5,"Face.name"]) {
  print(imgname)
  if (startsWith(imgname, "G")) {
    subdir = "Governors_all_stimuli"
  }
  else {
    subdir = "Senate_all_stimuli"
  }
  img = load.image(sprintf("%s/politicians-database_aligned/%s/%s.png", data_path, subdir, imgname))
  plot(img)
}
write.csv(sorted[,c("Face.name", "error", "Competency_Norm", "pred_Competent")], sprintf("%s/politicians-error.csv", output_path))

