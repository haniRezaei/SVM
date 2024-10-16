####Boosting and Support Vector Machines ####
library(ElemStatLearn)
data("SAheart")
colnames(SAheart)  

## Boosting ####
#install.packages('gbm')
library(gbm)
n<-nrow(SAheart)
set.seed(1234)
train<-sample(1:n,ceiling(n/2),replace = F)
heart_test<-SAheart[-train,]

boost.out<-gbm(chd~.,data=SAheart[train,],
               distribution="bernoulli",
               n.trees = 100,interaction.depth = 1,
               bag.fraction = 1)
boost.out

ntr<-length(train) # size of the training
heart_tr<-SAheart[train,] 
set.seed(1234)
folds<-sample(1:5,ntr,replace=T)
table(folds)
B<-seq(from=25,to=200,by=25)
err.cv<-matrix(NA,5,length(B),
               dimnames = list(NULL,paste0("B=",B[1:length(B)])))
for (i in 1:5){
  x.te<-heart_tr[folds==i,]
  x.tr<-heart_tr[folds!=i,]
  for (j in 1:length(B)){
    boost.out<-gbm(chd~.,x.tr,distribution="bernoulli",
                   bag.fraction=1,interaction.depth = 1,
                   n.trees = B[j])
    phat<-predict(boost.out,newdata = x.te,n.trees=B[j],
                  type="response")
    yhat<-ifelse(phat>0.5,1,0)
    err.cv[i,j]<-mean(yhat!=x.te$chd)
  }
}
err.cv
colMeans(err.cv)
b_best<-B[which.min(colMeans(err.cv))]

# Fit the best boosted trees on the whole training set
boost.train<-gbm(chd~.,data=SAheart[train,],distribution = "bernoulli",
                 n.trees = b_best,interaction.depth = 1,bag.fraction = 1)
phat.te<-predict(boost.train,newdata = SAheart[-train,],n.trees = b_best,
                 type="response")
yhat.te<-ifelse(phat.te>0.5,1,0)
table(yhat=yhat.te,SAheart$chd[-train])
mean(yhat.te!=SAheart$chd[-train])


## Support Vector Machines ####
library(e1071)

x<-SAheart[,-ncol(SAheart)]
y<-SAheart[,ncol(SAheart)]
heart.df<-data.frame(chd=as.factor(y),x)
?svm

out.svm<-svm(chd~.,data=heart.df,kernel="linear",cost=10)
summary(out.svm)

out.svm<-svm(chd~.,data=heart.df,kernel="linear",cost=0.1)
summary(out.svm)

### Support vector classifier ####
set.seed(1234)
tune_out<-tune(svm,chd~.,data=heart.df[train,],kernel="linear",
               ranges = list(cost=c(0.001,0.01,0.1,1,5,10,100)),
              tunecontrol=tune.control(cross=10))
# to change the no. of folds K -> cross=K
summary(tune_out)

best_model<-tune_out$best.model # storing the features of the best model
yhat.linear<-predict(best_model,newdata = heart.df[-train,])

table(yhat.linear,SAheart$chd[-train])
mean(yhat.linear!=SAheart$chd[-train])

### Support vector machine - Radial kernel ####
set.seed(1234)
tune_out<-tune(svm,chd~.,data=heart.df[train,],kernel="radial",
               ranges = list(cost=c(0.001,0.01,0.1,1,5,10,100,1000),
                             gamma=c(0.1,0.2,0.3,0.4,0.5)),
               tunecontrol=tune.control(cross=10))
# to change the no. of folds K -> cross=K
summary(tune_out)

best_model_radial<-tune_out$best.model # storing the features of the best model
yhat.radial<-predict(best_model_radial,newdata = heart.df[-train,])

table(yhat.radial,SAheart$chd[-train])
mean(yhat.radial!=SAheart$chd[-train])

## An attempt to see the effect of gamma 
# out.radial<-svm(chd~.,data=heart.df[train,],kernel="radial",
#                 gamma=1,cost=0.1)
# yyhat<-predict(out.radial,newdata = heart.df[-train,])
# table(yyhat,heart.df$chd[-train])


### Support Vector Machines - Polynomial kernel ####
set.seed(1234)
tune_out<-tune(svm,chd~.,data=heart.df[train,],kernel="polynomial",
               ranges = list(cost=c(0.001,0.01,0.1,1,5,10,100,1000),
                             gamma=c(0.1,0.2),
                             degree=1:5),
               tunecontrol=tune.control(cross=10))
# to change the no. of folds K -> cross=K
summary(tune_out)

best_model_polynomial<-tune_out$best.model # storing the features of the best model
yhat.polynomial<-predict(best_model_polynomial,newdata = heart.df[-train,])

table(yhat.polynomial,SAheart$chd[-train])
mean(yhat.polynomial!=SAheart$chd[-train])
