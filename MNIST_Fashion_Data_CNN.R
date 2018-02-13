################# DOWNLOAD PACKAGES #######################

rm(list=ls()) # Clear all variables

## Download and set up package "MXNet":
#cran <- getOption("repos")
#cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
#options(repos = cran)
#install.packages("mxnet")
library('mxnet')

################# UPLOAD DATA #####################

# Set: C:\Users\eagle\OneDrive\Documents\data_fashion
setwd("C:/Users/eagle/OneDrive/Documents/data_fashion")
data <- read.csv("train.csv")
all <- data; all[1:5,1:5]; dim(all)

###################### CONVOLUTION NEURAL NETWORK ########################

# Define number of convolution layers and NN layers 
# in the function. 

cnn <- function(
  # Tuning:
  batch.size = 40,
  alpha = 0.01,
  mom = 0.9,
  # Convolution layers:
  num_filter_1 = 128,
  num_filter_2 = 64,
  num_filter_3 = 50,
  num_filter_4 = 50,
  num_filter_5 = 50,
  # NN layers:
  # Parameters:
  a1 = 128, #128+4*256 # LeCun: 128
  a2 = 64, #64+4*64 # LeCun: 64
  a3 = 10, #64 # LeCun: 10
  a4 = 10,
  a5 = 10,
  iteration = 5,
  # Cutoff: 
  data.cutoff.line = 0.9 #(36/42)
)
{
  ################# SPLIT DATA ############################
  
  # Load train and test datasets
  train <- all[1:(data.cutoff.line*nrow(all)),]; dim(train)
  test <- all[(data.cutoff.line*nrow(all)+1):nrow(all),]; dim(test)
  
  # Set up train and test datasets
  train <- data.matrix(train)
  train_x <- t(train[, -1])
  train_y <- train[, 1]
  train_array <- train_x
  size <- round(sqrt(nrow(train_x)))
  dim(train_array) <- c(size, size, 1, ncol(train_x))
  
  test_x <- t(test[, -1])
  test_y <- test[, 1]
  test_array <- test_x
  dim(test_array) <- c(size, size, 1, ncol(test_x))
  
  #################### DESIGN CONVOLUTION LAYER #################
  
  # Set up the symbolic model
  data <- mx.symbol.Variable('data')
  # 1st convolutional layer
  conv_1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = num_filter_1) 
  tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
  pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
  # 2nd convolutional layer
  conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = num_filter_2) # LeCun: 50
  tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
  pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
  # 3rd convolutional layer
  conv_3 <- mx.symbol.Convolution(data = pool_2, kernel = c(3, 3), num_filter = num_filter_3)
  tanh_3 <- mx.symbol.Activation(data = conv_3, act_type = "tanh")
  pool_3 <- mx.symbol.Pooling(data=tanh_3, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
  # 4th convolutional layer
  conv_4 <- mx.symbol.Convolution(data = pool_3, kernel = c(1, 1), num_filter = num_filter_4)
  tanh_4 <- mx.symbol.Activation(data = conv_4, act_type = "tanh")
  pool_4 <- mx.symbol.Pooling(data=tanh_4, pool_type = "max", kernel = c(1, 1), stride = c(1, 1))
  # 5th convolutional layer
  conv_5 <- mx.symbol.Convolution(data = pool_4, kernel = c(1, 1), num_filter = num_filter_5)
  tanh_5 <- mx.symbol.Activation(data = conv_5, act_type = "tanh")
  pool_5 <- mx.symbol.Pooling(data=tanh_5, pool_type = "max", kernel = c(1, 1), stride = c(1, 1))
  # 1st fully connected layer
  
  # CHOOSE HOW MANY CONV LAYERS?
  flatten <- mx.symbol.Flatten(data = pool_2) # Watch which pool_i to use
  fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = a1) # LeCun: 500 
  tanh_6 <- mx.symbol.Activation(data = fc_1, act_type = "tanh")
  # 2nd fully connected layer
  fc_2 <- mx.symbol.FullyConnected(data = tanh_6, num_hidden = a2) # LeCun: 40
  tanh_7 <- mx.symbol.Activation(data = fc_2, act_type = "tanh")
  # 3rd fully connected layer
  fc_3 <- mx.symbol.FullyConnected(data = tanh_7, num_hidden = a3)
  tanh_8 <- mx.symbol.Activation(data = fc_3, act_type = "tanh")
  # 4th fully connected layer
  fc_4 <- mx.symbol.FullyConnected(data = tanh_8, num_hidden = a4)
  tanh_9 <- mx.symbol.Activation(data = fc_4, act_type = "tanh")
  # 5th fully connected layer
  fc_5 <- mx.symbol.FullyConnected(data = tanh_9, num_hidden = a5)
  
  # CHOOSE HOW MANY NN LAYERS?
  # Output. Softmax output since we'd like to get some probabilities.
  NN_model <- mx.symbol.SoftmaxOutput(data = fc_3)
  
  
  #################### TRAINING ##################################
  
  # Pre-training set up: 
  # Set seed for reproducibility
  mx.set.seed(100)
  
  # Device used. CPU in my case.
  devices <- mx.cpu()
  
  # Training
  iter <- iteration
  
  # Train the model
  model <- mx.model.FeedForward.create(
    NN_model, 
    X = train_array,
    y = train_y,
    ctx = devices,
    num.round = iter, # LeCun: 480
    array.batch.size = batch.size,
    learning.rate = alpha, # LeCun: 0.01
    momentum = mom, # LeCun: 0.9
    eval.metric = mx.metric.accuracy,
    epoch.end.callback = mx.callback.log.train.metric(100)
  )
  
  ###################### PREDICTION #########################
  
  # Testing:
  # Predict labels
  predicted <- predict(model, test_array)
  # Assign labels
  predicted_labels <- max.col(t(predicted)) - 1
  
  table <- table(predicted_labels, test_y)
  percent <- sum(diag(table(predicted_labels, test_y)))/sum(table(predicted_labels, test_y))
  # percent; 1-percent
  
  # Final results:
  return(
    list(
      prediction.table = table,
      Testing.Accuracy = percent,
      Testing.Error = 1 - percent
    )
  )
} # End of function.

# test:
#cnn(iteration = 17)

###################### TUNING CNN: Function ########################

# Create a tuning neural network function:
tune.cnn <- function(
  # Default:
  default.a1 = 128,
  default.a2 = 64,
  default.a3 = 10
) {
  
  # Default:
  #default.a1 = 128
  #default.a2 = 64
  #default.a3 = 2
  
  # Tune the number of iterations:
  tune.iter <- NULL
  tune.iter.interval <- seq(10,90,30)
  for (i in c(tune.iter.interval)) {
    tune.iter <- cbind(tune.iter,cnn(
      #batch.size = 40,
      #alpha = 0.01,
      #mom = 0.9,
      a1 = default.a1,
      a2 = default.a2,
      a3 = default.a3,
      iter = i)$Testing.Accuracy
    )
    print(c("Finished tuning iter with rounds", i))
  }; tune.iter; tune.iter.interval
  tune.iter.mat <- data.frame(rbind(
    tune.iter.interval, tune.iter
  )); t(tune.iter.mat)
  print("Finished tuning iterations!")
  inds.iter <- which(tune.iter.mat == max(tune.iter), arr.ind = TRUE)
  i <- tune.iter.interval[inds.iter[1,2]]
  
  # Tune the number of hidden neurons in the 1st hidden layer:
  tune.a1 <- NULL
  tune.a1.interval <- seq(128,128*12,128)
  for (j in c(tune.a1.interval)) {
    tune.a1 <- cbind(tune.a1,cnn(
      #batch.size = 40,
      #alpha = 0.01,
      #mom = 0.9,
      a1 = j,
      a2 = 64,
      a3 = 10,
      iter = i)$Testing.Accuracy
    )
    print(c("Finished tuning a1 with rounds", j))
  }; tune.a1; tune.a1.interval
  tune.a1.mat <- data.frame(rbind(
    tune.a1.interval, tune.a1
  )); t(tune.a1.mat)
  print("Finished tuning the first hidden layer!")
  inds.a1 <- which(tune.a1.mat == max(tune.a1), arr.ind = TRUE)
  j <- tune.a1.interval[inds.a1[1,2]]
  
  # Tune the number of hidden neurons in the 2nd hidden layer:
  tune.a2 <- NULL
  tune.a2.interval <- seq(64,64*12,64)
  for (k in c(tune.a2.interval)) {
    tune.a2 <- cbind(tune.a2,cnn(
      #batch.size = 40,
      #alpha = 0.01,
      #mom = 0.9,
      a1 = j,
      a2 = k,
      a3 = 10,
      iter = i)$Testing.Accuracy
    )
    print(c("Finished tuning a2 with rounds", k))
  }; tune.a2; tune.a2.interval
  tune.a2.mat <- data.frame(rbind(
    tune.a2.interval, tune.a2
  )); t(tune.a2.mat)
  print("Finished tuning the second hidden layer!")
  inds.a2 <- which(tune.a2.mat == max(tune.a2), arr.ind = TRUE)
  k <- tune.a2.interval[inds.a2[1,2]]
  
  # Tune the number of hidden neurons in the 3rd hidden layer:
  tune.a3 <- NULL
  tune.a3.interval <- c(10,15,20)
  for (m in c(tune.a3.interval)) {
    tune.a3 <- cbind(tune.a3,cnn(
      #batch.size = 40,
      #alpha = 0.01,
      #mom = 0.9,
      a1 = j,
      a2 = k,
      a3 = m,
      iter = i)$Testing.Accuracy
    )
    print(c("Finished tuning a3 with rounds", m))
  }; tune.a3; tune.a3.interval
  tune.a3.mat <- data.frame(rbind(
    tune.a3.interval, tune.a3
  )); t(tune.a3.mat)
  print("Finished tuning the third hidden layer!")
  inds.a3 <- which(tune.a3.mat == max(tune.a3), arr.ind = TRUE)
  m <- tune.a3.interval[inds.a3[1,2]]
  
  # Tune the number of iterations again:
  tune.iter.again <- NULL
  tune.iter.interval.again <- seq(70,120,10)
  for (i in c(tune.iter.interval.again)) {
    tune.iter.again <- cbind(tune.iter.again,cnn(
      #batch.size = 40,
      #alpha = 0.01,
      #mom = 0.9,
      a1 = j,
      a2 = k,
      a3 = m,
      iter = i)$Testing.Accuracy
    )
    print(c("Finished tuning iter again with rounds", i))
  }; tune.iter.again; tune.iter.interval.again
  tune.iter.again.mat <- data.frame(rbind(
    tune.iter.interval.again, tune.iter.again
  )); t(tune.iter.again.mat)
  print("Finished tuning number of iterations again!")
  inds.iter.again <- which(tune.iter.again.mat == max(tune.iter.again), arr.ind = TRUE)
  i <- tune.iter.interval.again[inds.iter.again[1,2]]
  #beep(sound=10) # Finishing alert!
  
  return(list(
    i,j,k,m
  ))
} # Finish tuning function

###################### TUNING CNN: Run it ########################

# Tune by using function **tune.nn()**
output <- tune.cnn(
  default.a1 = 128,
  default.a2 = 64,
  default.a3 = 10
); output

# Extract output:
i <- as.numeric(output[1]) # Iteration
j <- as.numeric(output[2]) # Number of hidden neurons for 1st hidden layer
k <- as.numeric(output[3]) # Number of hidden neurons for 2nd hidden layer
m <- as.numeric(output[4]) # Number of hidden neurons for 3rd hidden layer


cnn(a1 = j, a2 = k, a3 = m, iter = i)