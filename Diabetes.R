df <- read.csv("New folder/R_Projects/diabetes.csv")
head(df)
summary(df)

missing_data <- sum(is.na(df))
print(paste("Total Missing Values:", missing_data))

str(df)

cor_matrix <- cor(df[,1:8])  # Excluding the Outcome variable
cor_matrix

library(ggplot2)
ggplot(df, aes(x=Glucose)) + geom_histogram(binwidth=10, fill="blue", color="black") + 
  labs(title="Distribution of Glucose Levels")

ggplot(df, aes(x=factor(Outcome), y=Glucose)) + 
  geom_boxplot(fill="orange") +
  labs(title="Boxplot of Glucose vs Outcome")

pairs(df[,1:8], col=df$Outcome)

table(df$Outcome)
prop.table(table(df$Outcome)) * 100

library(corrplot)
corrplot(cor(df[,1:8]), method="circle")

t_test_result <- t.test(df$Glucose ~ df$Outcome)
t_test_result

model <- glm(Outcome ~ ., data=df, family=binomial)
summary(model)

predicted <- ifelse(predict(model, df, type="response") > 0.5, 1, 0)
accuracy <- mean(predicted == df$Outcome)
print(paste("Model Accuracy:", accuracy))

coef_df <- as.data.frame(coef(model))
barplot(coef_df$`coef(model)`, names.arg=rownames(coef_df), las=2, col="skyblue")
