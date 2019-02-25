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

