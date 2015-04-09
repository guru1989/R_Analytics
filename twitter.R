# Twitter sentiment classification using a streaming model in R.


install.packages("streamR")
install.packages("twitteR")
install.packages("ROAuth")
install.packages("tm")
library(streamR)
library(ROAuth)
library(twitteR)
library(tm)
install.packages("devtools")
library("devtools")
install.packages("ff")
install.packages("rJava")
install_github("jwijffels/RMOA", subdir="RMOAjars/pkg")
install_github("jwijffels/RMOA", subdir="RMOA/pkg")
require("RMOA")


requestURL <- "https://api.twitter.com/oauth/request_token"
accessURL <- "https://api.twitter.com/oauth/access_token"
authURL <- "http://api.twitter.com/oauth/authorize"
consumerKey <- ""
consumerSecret <- ""


# Data Retrieval =============================================================================================== 

# User authentication
my_oauth <- OAuthFactory$new(consumerKey=consumerKey,
                             consumerSecret=consumerSecret, requestURL=requestURL,
                             accessURL=accessURL, authURL=authURL)

# Please enter the authetication key that you get from the web server after running this command
my_oauth$handshake(cainfo = system.file("CurlSSL", "cacert.pem", package = "RCurl"))

registerTwitterOAuth(my_oauth)


# Build the training data with around 5000 tweets filtered for keywords "love" & "hate"
filterStream("train.json", track = c("love","hate"), tweets = 5000,lang="en",
             oauth = my_oauth)



tweets_train=parseTweets("train.json", simplify = TRUE)

file.remove("train.json")

# Data Retrieval ends ========================================================================================


# Data Pre-processing ========================================================================================

# Preprocessing the data - remove punctuation, stopwords, stemming etc...
twtr <- Corpus(VectorSource(tweets_train$text))
twtr = tm_map(twtr,stripWhitespace)
twtr <- tm_map(twtr, content_transformer(tolower))
twtr <- tm_map(twtr, removePunctuation)
twtr <- tm_map(twtr, removeWords, stopwords("english"))
twtr <- tm_map(twtr, removeNumbers)
twtr <- tm_map(twtr, stemDocument)

# Data Pre-processing ends ====================================================================================


# Feature Construction & Feature Selection ====================================================================

# Construct the DocumentTermMatrix and prune the attributes having sparsity >99%
myDtmtr=DocumentTermMatrix(twtr)
myDtmtr=removeSparseTerms(myDtmtr, 0.99)
t=as.matrix(myDtmtr)



# Create a new column called class with value 1 if it contains love and 0 if hate
class=rep(0,dim(t)[1])
t=cbind(t,class)
d=data.frame(t)


# Remove sentences containing both love and hate in the same tweet
d=d[!(d$love>=1 & d$hate>=1),]
d[d$love>=1,]$class=1

# Remove the columns love & hate
d=subset(d, select=-hate)
d=subset(d, select=-love)
d$class=as.factor(d$class)


# Feature Construction & Feature Selectio ends ====================================================================


# Dynamic Model Building =======================================================================================

# Apply the Hoeffding tree model and predict training outcome and get confusion matrix
hdt <- HoeffdingTree(splitConfidence="1e-1")
datastream <- datastream_dataframe(data=d)
model <- trainMOA(model=hdt, formula= class~., data=datastream, reset=TRUE, trace=TRUE)
pred=predict(model,newdata=subset(d, select=-class),type="response")
table(pred,d$class)
acc_table=table(pred,d$class)

# Print the training accuracy
train_accuracy=(acc_table[1,1]+acc_table[2,2])/sum(acc_table)
cat("Training accuracy=",train_accuracy,"\n")


# Dynamic Model Building ends =======================================================================================


# Dynamic Model Update  & Prediction==========================================================================================

# Construct the test data with about 100 tweets filtered for keywords "love" & "hate"
filterStream("test.json", track = c("love","hate"), tweets = 100,lang="en",
             oauth = my_oauth)

tweets_test=parseTweets("test.json", simplify = TRUE)

file.remove("test.json")

# Preprocessing the data - remove punctuation, stopwords, stemming etc...
twte <- Corpus(VectorSource(tweets_test$text))
twte = tm_map(twte,stripWhitespace)
twte <- tm_map(twte, content_transformer(tolower))
twte <- tm_map(twte, removePunctuation)
twte <- tm_map(twte, removeWords, stopwords("english"))
twte <- tm_map(twte, removeNumbers)
twte <- tm_map(twte, stemDocument)

# Construct the DocumentTermMatrix
myDtmte <- DocumentTermMatrix(twte)
t <- as.matrix(myDtmte)


# Create new column "class" whose value is 1 if the tweet contains keyword "love" and 0 if it contains "hate"
# This is to update the model with the correct class after prediction
class=rep(0,dim(t)[1])
t=cbind(t,class)
test=data.frame(t)
test[test$love>=1,]$class=1

# Remove the columns love & hate
test=subset(test, select=-hate)
test=subset(test, select=-love)
test$class=as.factor(test$class)



temp=d[1,]
for (i in 1:ncol(temp))
{
  temp[1,i]=0
}

# Loop to parse each test tweet, take into account only attributes present in test data and update the model as well as predict the results.

pred_list=c()

for (i in 1:nrow(test))
{
  print(tweets_test$text[i])
  for (j in 1 :ncol(temp))
  {
    if(!is.null(test[i,colnames(temp)[j]]))
    {
      temp[1,j]=test[i,colnames(temp)[j]]
    }
  }
  
  pred=predict(model,newdata=subset(temp, select=-class),type="response")
  pred_list=c(pred_list,pred)
  
  temp[1,"class"]=as.factor(test[i,"class"])
  datastream <- datastream_dataframe(data=temp)
  model <- trainMOA(model=hdt, formula= class~., data=datastream, reset=FALSE)
  
  if(pred=="0")
    res="hate"
  else
    res="love"
  
  
  cat("predicted class = ",res,"\n")
  for (i in 1:ncol(temp))
  {
    temp[1,i]=0
  }
  
  
}


# Calculate the test accuracy
pred_list=as.factor(pred_list)

hits=0
tc=0
for(i in 1:length(test$class))
{
  if(test$class[i]==pred_list[i])
  {
    hits=hits+1
    tc=tc+1
  }
  else
    tc=tc+1
}
test_accuracy=hits/tc
cat("Test accuracy=",test_accuracy,"\n")

# Dynamic Model Update  & Prediction ends ==========================================================================================



