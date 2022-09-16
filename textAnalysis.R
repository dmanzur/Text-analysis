## reset
rm(list = ls())
gc()
## libraries
library(qdap)
library(sentimentr)
library(tm)         # main text mining library
library(SnowballC)  # used for word stemming
library(textstem)   #used for steaming
library(wordcloud)
library(reshape2)
library(text2vec)
library(e1071) # for caret
library(tidyr) # for pivot_wider
library(MLmetrics)
library(ggplot2)
library(caret) 
library(stringr)
library(xgboost)
library(dplyr)

getwd()
## read data
fileReviews.df<- read.csv("reviews.csv")

# Take one app only
fileReviews.df <- fileReviews.df[(fileReviews.df$appId)=="com.oristats.habitbull", ]

## Remove unnecessary columns and missing values
fileReviews.df <- subset(fileReviews.df, select = -c(reviewCreatedVersion,appId,at,userName,reviewId,userImage,sortOrder,thumbsUpCount,replyContent,repliedAt))
fileReviews.df$content <- iconv(fileReviews.df$content, "latin1", "ASCII", sub="")

###Corpus levels Preprocess text data
preprocess_Corpus <- function(reviews){
  myCorpus <- VCorpus(VectorSource(reviews)) # make to corpus
  myCorpus <- tm_map(myCorpus, stripWhitespace) # remove white space
  myCorpus <- tm_map(myCorpus, content_transformer(tolower)) # lower case
  myCorpus <- tm_map(myCorpus, removeNumbers) #remove numbers
  myCorpus <- tm_map(myCorpus, removePunctuation) # remove punctuation
  myCorpus <- tm_map(myCorpus, removeWords, stopwords("english"))
  myCorpus <- tm_map(myCorpus, stemDocument) #stem words
  return(myCorpus)
}

all_reviews_Corpus <- preprocess_Corpus(fileReviews.df$content) 
fileReviews.df$content <- sapply(all_reviews_Corpus, as.character)

# learn about the app scores
scoresTableDf <- as.data.frame(table(fileReviews.df$score))
colnames(scoresTableDf)<- c("score","count")
ggplot(scoresTableDf,aes(x=score,y=count))+geom_col()
table(fileReviews.df$score)
table(fileReviews.df$score)/nrow(fileReviews.df) # in %
rm(all_reviews_Corpus)
gc()


########## Vizzz

##first avgSentiment density
sentiment_content <- sentiment_by(fileReviews.df$content)
fileReviews.df$sentiment_content = sentiment_content$ave_sentiment

#density and avg_sentiment
ggplot(sentiment_content, aes(x=ave_sentiment, y= ..density..)) + 
  geom_density(aes(colour = as.factor(fileReviews.df$score)), size=1, alpha=.3) 

#separate the df to 5 (create separate tables by score)
fileReviews_df_1 <- fileReviews.df[fileReviews.df$score == 1,]
fileReviews_df_2 <- fileReviews.df[fileReviews.df$score == 2,]
fileReviews_df_3 <- fileReviews.df[fileReviews.df$score == 3,]
fileReviews_df_4 <- fileReviews.df[fileReviews.df$score == 4,]
fileReviews_df_5 <- fileReviews.df[fileReviews.df$score == 5,]


## text into corpus
reviews_1_corpus <- preprocess_Corpus(fileReviews_df_1$content) 
reviews_2_corpus <- preprocess_Corpus(fileReviews_df_2$content) 
reviews_3_corpus <- preprocess_Corpus(fileReviews_df_3$content) 
reviews_4_corpus <- preprocess_Corpus(fileReviews_df_4$content) 
reviews_5_corpus <- preprocess_Corpus(fileReviews_df_5$content) 


## TFxIDF in @1 & @2 & 3  & @4 & 5 
combinedDf <- c(paste(fileReviews_df_1$content, collapse = " "),
                paste(fileReviews_df_2$content, collapse = " "),
                paste(fileReviews_df_3$content, collapse = " "),
                paste(fileReviews_df_4$content, collapse = " "),
                paste(fileReviews_df_5$content, collapse = " "))
combinedCorpus <- preprocess_Corpus(combinedDf)
combinedDtm <- DocumentTermMatrix(combinedCorpus, control=list(weighting=weightTfIdf))

combinedDtm <- DocumentTermMatrix(combinedCorpus)
combinedDtm.m <- t(as.matrix(combinedDtm))
colnames(combinedDtm.m) <- c("1", "2","3","4", "5")

##Second - comparison cloud
comparison.cloud(combinedDtm.m,
                 max.words = 300,
                 random.order = FALSE,
                 scale = c(2, 0.5),
                 colors = c('black','#5c4a72', '#d6617f', '#b9c406', '#0878a4'))

# commonality cloud
commonality.cloud(combinedDtm.m,
                  max.words = 300,
                  random.order = FALSE,
                  colors = c('black','#5c4a72', '#d6617f', '#b9c406', '#0878a4'))


#Third - Term Frequency
#score 1
score_1_TermFreq <- as.matrix(combinedDtm)[1,]
topTerms <- score_1_TermFreq[order(score_1_TermFreq, decreasing = T)][1:20]
topTermsDf <- data.frame(term = names(topTerms), frequency = topTerms)
ggplot(topTermsDf, aes(x = reorder(term, frequency), y = frequency)) +
  geom_bar(stat="identity", fill='darkred') +
  coord_flip() +
  xlab("term score 1")

#score 2
score_2_TermFreq <- as.matrix(combinedDtm)[2,]
topTerms <- score_2_TermFreq[order(score_1_TermFreq, decreasing = T)][1:20]
topTermsDf <- data.frame(term = names(topTerms), frequency = topTerms)
ggplot(topTermsDf, aes(x = reorder(term, frequency), y = frequency)) +
  geom_bar(stat="identity", fill='darkred') +
  coord_flip() +
  xlab("term score 2")

#score 3
score_3_TermFreq <- as.matrix(combinedDtm)[3,]
topTerms <- score_3_TermFreq[order(score_3_TermFreq, decreasing = T)][1:20]
topTermsDf <- data.frame(term = names(topTerms), frequency = topTerms)
ggplot(topTermsDf, aes(x = reorder(term, frequency), y = frequency)) +
  geom_bar(stat="identity", fill='darkred') +
  coord_flip() +
  xlab("term score 3")

#score 4
score_4_TermFreq <- as.matrix(combinedDtm)[4,]
topTerms <- score_4_TermFreq[order(score_4_TermFreq, decreasing = T)][1:20]
topTermsDf <- data.frame(term = names(topTerms), frequency = topTerms)
ggplot(topTermsDf, aes(x = reorder(term, frequency), y = frequency)) +
  geom_bar(stat="identity", fill='darkred') +
  coord_flip() +
  xlab("term score 4")

#score 5
score_5_TermFreq <- as.matrix(combinedDtm)[5,]
topTerms <- score_5_TermFreq[order(score_5_TermFreq, decreasing = T)][1:20]
topTermsDf <- data.frame(term = names(topTerms), frequency = topTerms)
ggplot(topTermsDf, aes(x = reorder(term, frequency), y = frequency,fill=frequency)) +
  geom_bar(stat="identity", fill='darkred') +
  coord_flip() +
  xlab("term score 5")

##top terms score
score_TermFreq <- as.data.frame(as.matrix(combinedDtm)[1:5,])
score_TermFreq$score <- as.numeric(row.names(score_TermFreq))
topTermsDf <- data.frame(term = names(topTerms), frequency = topTerms)

top_terms <- score_TermFreq[,c(topTermsDf$term,'score')]
top_terms1 <- stack(top_terms)
top_terms1$score <- top_terms$score
top_terms1 <- head(top_terms1,100)
colnames(top_terms1)<- c("frequency","word","score")
# Grouped
ggplot(top_terms1, aes(fill=as.factor(score), y=frequency, x=word)) + 
  geom_bar(position="dodge", stat="identity")

###############################################
### Models ###

set.seed(1)

#remove 3 common words: app, habit, use
tempCorpus <- VCorpus(VectorSource(fileReviews.df$content))
fileReviews.df$content <- sapply(tm_map(tempCorpus, removeWords, c("app","use","habit")), as.character)

trainIDs <- sample(1:(dim(fileReviews.df)[1]), dim(fileReviews.df)[1]*0.7)
train <- fileReviews.df[trainIDs,] 
valid <- fileReviews.df[-trainIDs,]

train.Corpus = preprocess_Corpus(train$content)
valid.Corpus = preprocess_Corpus(valid$content)


#show Residuals Results
showResidualsPlot <- function(predsValues,originalValues){
  predsDF <- data.frame(predsValues,originalValues)
  x = 1:length(originalValues)
  r <- originalValues - predsValues
  plot(r, ylab = "residuals", main = "Residuals")
  lines(lowess(x, r), col = "blue",lwd = 4)
}

# BoW + sentiment---------------------------------------------------------------------
train.dtm <- DocumentTermMatrix(train.Corpus)
train.dtm <- removeSparseTerms(train.dtm, .95)

valid.dtm <- DocumentTermMatrix(valid.Corpus, control=list(dictionary=train.dtm$dimnames$Terms))

train.df <- data.frame(y = train$score, as.matrix(train.dtm))
valid.df <- data.frame(y = valid$score, as.matrix(valid.dtm))

train.df$sentiment_score <-train$sentiment_content 
valid.df$sentiment_score <-valid$sentiment_content

train.df <- na.omit(train.df) 
valid.df <- na.omit(valid.df) 
valid.df <- valid.df[order(valid.df$y),]


#linear regression
lm.model <- lm(y ~ ., data = train.df) 
lm.step.reg <- step(lm.model, direction = "backward")
preds.lm <-predict(lm.step.reg,newdata = valid.df)
RMSE(preds.lm,valid.df$y)
MAPE(preds.lm, valid.df$y)

#XGBOOST
xgb_train = xgb.DMatrix(data = as.matrix(train.df[,-1]), label = train.df[,1])
xgb_valid = xgb.DMatrix(data = as.matrix(valid.df[,-1]), label = valid.df[,1])
model_xgboost = xgboost(data = xgb_train, max.depth = 3, nrounds = 86, verbose = 0)
preds.xgb = predict(model_xgboost, xgb_valid)
RMSE(preds.xgb,valid.df$y)
MAPE(preds.xgb, valid.df$y)


# word2vec + sentiment----------------------------------------------------------------
it <- itoken(train$content)
v <- create_vocabulary(it)
vectorizer <- vocab_vectorizer(v)

tcm <- create_tcm(it, vectorizer, skip_grams_window = 10L,)
glove <- GlobalVectors$new(rank = 50, x_max = 10) ### change
word_context <- glove$fit_transform(tcm, n_iter = 50)
word_vectors <- glove$components + t(word_context)

train.word2vec <- create_dtm(it, vectorizer)
train.word2vec <- as.matrix((train.word2vec/Matrix::rowSums(train.word2vec)) %*% t(word_vectors))

valid.it <- itoken(valid$content)
valid.word2vec <- create_dtm(valid.it, vectorizer)
valid.word2vec <- as.matrix((valid.word2vec/Matrix::rowSums(valid.word2vec)) %*% t(word_vectors))

word2vec.train.df <- data.frame(y = train$score, train.word2vec)
word2vec.valid.df <- data.frame(y = valid$score, valid.word2vec)

word2vec.train.df <- na.omit(word2vec.train.df) 
word2vec.valid.df <- na.omit(word2vec.valid.df) 

word2vec.valid.df <- word2vec.valid.df[order(word2vec.valid.df$y),]

#linear regression
lm.word2vec.model <- lm(y ~ ., data = word2vec.train.df) 
lm.word2vec.step.reg <- step(lm.word2vec.model, direction = "backward")
preds.lm.word2vec <-predict(lm.word2vec.step.reg,newdata = word2vec.valid.df)
RMSE(preds.lm.word2vec,word2vec.valid.df$y)
MAPE(preds.lm.word2vec, word2vec.valid.df$y)

#XGBoost
xgb_train_word2vec = xgb.DMatrix(data = as.matrix(word2vec.train.df[,-1]), label = word2vec.train.df[,1])
xgb_valid_word2vec = xgb.DMatrix(data = as.matrix(word2vec.valid.df[,-1]), label = word2vec.valid.df[,1])
model_xgboost_word2vec = xgboost(data = xgb_train_word2vec, max.depth = 3, nrounds = 86, verbose = 0)

preds.xgb.word2vec = predict(model_xgboost_word2vec, xgb_valid_word2vec)
RMSE(preds.xgb.word2vec,word2vec.valid.df$y)
MAPE(preds.xgb.word2vec, word2vec.valid.df$y)

showResidualsPlot(preds.xgb.word2vec,word2vec.valid.df$y)
hist(preds.xgb.word2vec,breaks = 20) 
plot(as.factor(word2vec.valid.df$y))

# BoW + text2vec + sentiment ----------------------------------------------------------

train.combine <- data.frame(as.matrix(train.dtm), as.matrix(train.word2vec))
valid.combine <- data.frame(as.matrix(valid.dtm), as.matrix(valid.word2vec))

combine.train.df <- data.frame(y = train$score, train.combine)
combine.valid.df <- data.frame(y = valid$score, valid.combine)

combine.train.df$sentiment_score <-train$sentiment_content 
combine.valid.df$sentiment_score <-valid$sentiment_content

combine.train.df <- na.omit(combine.train.df) 
combine.valid.df <- na.omit(combine.valid.df) 

combine.valid.df <- combine.valid.df[order(combine.valid.df$y),]

#linear regression
lm.combine.model <- lm(y ~ ., data = train.df) 
lm.combine.step.reg <- step(lm.combine.model, direction = "backward")
preds.lm.combine <-predict(lm.combine.step.reg,newdata = combine.valid.df)
RMSE(preds.lm.combine,combine.valid.df$y)
MAPE(preds.lm.combine, combine.valid.df$y)


#XGBOOST
xgb_train = xgb.DMatrix(data = as.matrix(combine.train.df[,-1]), label = combine.train.df[,1])
xgb_valid = xgb.DMatrix(data = as.matrix(combine.valid.df[,-1]), label = combine.valid.df[,1])
model_xgboost = xgboost(data = xgb_train, max.depth = 3, nrounds = 86, verbose = 0)
preds.xgb.combine = predict(model_xgboost, xgb_valid)
RMSE(preds.xgb.combine,combine.valid.df$y)
MAPE(preds.xgb.combine, combine.valid.df$y)

#show results in Viz
showResidualsPlot(preds.xgb.combine,combine.valid.df$y)
hist(preds.xgb.combine,breaks = 20) 
plot(as.factor(combine.valid.df$y))

