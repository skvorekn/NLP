# https://www.r-bloggers.com/how-to-prepare-data-for-nlp-text-classification-with-keras-and-tensorflow/

# convert data into numeric
# install_keras()
# library(keras)
# library(tidyverse)

devtools::install_github("rstudio/tensorflow")
library(tensorflow)
install_tensorflow(method = "conda", conda = "auto",
                   version = "1.5.0", envname = "r-tensorflow")
library(reticulate)
use_condaenv("r-tensorflow", required = TRUE)
library(keras)
install_keras(method = "conda", tensorflow = "1.5.0")
is_keras_available()
library(readr)
library(dplyr)
library(ggplot2)

# data = text reviews of different items of clothing + rating, division
# classify whether or not the item was liked using rating to train

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

clothing_reviews <- read_csv("Womens Clothing E-Commerce Reviews.csv") %>%
  mutate(Liked = ifelse(Rating == 5, 1, 0),
         text = paste(Title, `Review Text`),
         text = gsub("NA", "", text))
head(clothing_reviews)

glimpse(clothing_reviews)

clothing_reviews %>%
  ggplot(aes(x = factor(Liked), fill = Liked)) +
  geom_bar(alpha = 0.8) +
  guides(fill = FALSE)

# tokenize the text

text <- clothing_reviews$text

# define number of words we want to consider = feature space
# use 1000 most frequent words
max_features <- 1000
tokenizer <- keras::text_tokenizer(num_words = max_features)
tokenizer

# fit the tokenizer to our text data
tokenizer %>% 
  fit_text_tokenizer(text)

tokenizer$document_count

tokenizer$word_index %>%
  head()

# excludes words not in the top 1000
text_seqs <- texts_to_sequences(tokenizer, text)

text_seqs %>%
  head()

# Set parameters:
maxlen <- 100
batch_size <- 32
embedding_dims <- 50
filters <- 64
kernel_size <- 3
hidden_dims <- 50
epochs <- 5

# return a matrix with # columns = # words (or the # words in the longest sentence)
# reviews with fewer words are padded with 0s
# longer reviews are cut
x_train <- text_seqs %>%
  pad_sequences(maxlen = maxlen)
dim(x_train)

# encode response variables with 1 for 5 stars and 0 for anything else
y_train <- clothing_reviews$Liked
length(y_train)

# now need to convert into something that will give info about the features
# word embeddings or word vectors are learned from the text data
# encode in a few dimensions while maximizing the information
# could also use one-hot encoding,  one-hot hashing, or pre-trained embeddings (GloVe)

model <- keras_model_sequential() %>% 
  layer_embedding(max_features, embedding_dims, input_length = maxlen) %>%
  layer_dropout(0.2) %>%
  layer_conv_1d(
    filters, kernel_size, 
    padding = "valid", activation = "relu", strides = 1
  ) %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(hidden_dims) %>%
  layer_dropout(0.2) %>%
  layer_activation("relu") %>%
  layer_dense(1) %>%
  layer_activation("sigmoid") %>% compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )
model
hist <- model %>%
  fit(
    x_train,
    y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.3
  )
plot(hist)
