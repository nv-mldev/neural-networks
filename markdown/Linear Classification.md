
binary classification, where
the goal is to predict a binary-valued target. Here are some examples of
binary classification problems:
• You want to train a medical diagnosis system to predict whether a
patient has a given disease. You have a training set consisting of a
set of patients, a set of features for those individuals (e.g. presence or
absence of various symptoms), and a label saying whether or not the
patient had the disease.
• You are running an e-mail service, and want to determine whether
a given e-mail is spam. You have a large collection of e-mails which
have been hand-labeled as spam or non-spam.
• You are running an online payment service, and want to determine
whether or not a given transaction is fraudulent. You have a labeled
training dataset of fraudulent and non-fraudulent transactions; fea-
tures might include the type of transaction, the amount of money, or
the time of day

Like regression, binary classification is a very restricted kind of task.
Most learning problems you’ll encounter won’t fit nicely into one of these
two categories. Our motivation for focusing on binary classification is to
introduce several fundamental ideas that we’ll use throughout the course.
In this lecture, we discuss how to view both data points and linear classifiers
as vectors. Next lecture, we discuss the perceptron, a particular classifica-
tion algorithm, and use it as an example of how to efficiently implement a
learning algorithm in Python.

This lecture focuses on the geometry of classification. We’ll look in
particular at two spaces:
• The input space, where each data case corresponds to a vector. A
classifier corresponds to a decision boundary, or a hyperplane such
that the positive examples lie on one side, and negative examples lie
on the other side.

Weight space, where each set of classification weights corresponds to
a vector. Each training case corresponds to a constraint in this space,
where some regions of weight space are “good” (classify it correctly)
and some regions are “bad” (classify it incorrectly).

Learning goals
• Know what is meant by binary linear classification.
• Understand why an explicit threshold for a classifier is redundant.
Understand how we can get rid of the bias term by adding a “dummy”
feature.
• Be able to specify weights and biases by hand to represent simple
functions (e.g. AND, OR, NOT).
• Be familiar with input space and weight space.
– Be able to plot training cases and classification weights in both
input space and weight space.
• Be aware of the limitations of linear classifiers.
– Know what is meant by convexity, and be able to use convexity
to show that a given set of training cases is not linearly separable.
– Understand how we can sometimes still separate the classes using
a basis function representation.

## Binary Linear Classifiers

 at classifiers which are both binary (they distinguish be-
tween two categories) and linear (the classification is done using a linear
function of the inputs). As in our discussion of linear regression, we assume
each input is given in terms of D scalar values, called input dimensions
or features, which we think summarize the important information for clas-
sification. (Some of the features, e.g. presence or absence of a symptom,
may in fact be binary valued, but we’re going to treat these as real-valued
(i)
anyway.) The jth feature for the ith training example is denoted x_{j}^(i) . All
of the features for a given training case are concatenated together to form a
vector, which we’ll denote x^(i) .

Associated with each data case is a binary-valued target, the thing we’re
trying to predict. By definition, a binary target takes two possible values,

which we’ll call classes, and which are typically referred to as positive
and negative. (E.g., the positive class might be “has disease” and the
negative class might be “does not have disease.”) Data cases belonging
to these classes are called positive examples and negative examples,
respectively.

 The training set consists of a set of N pairs (x(i) , t(i) ), where
x(i) is the input and t(i) is the binary-valued target, or label. Since the
training cases come with labels, they’re referred to as labeled examples.
Confusingly, even though we talk about positive and negative examples, the
t(i) typically take values in {0, 1}, where 0 corresponds to the “negative”
class.

Our goal is to correctly classify all the training cases (and, hopefully,
examples not in the training set). In order to do the classification, we need
to specify a model, which determines how the predictions are computed
from the inputs. As we said before, our model for this week is binary linear
classifiers.
The way binary linear classifiers work is simple: they compute a linear
function of the inputs, and determine whether or not the value is larger
than some threshold r. Recall from Lecture 2 that a linear function of the
input can be written as :

Threshold and Biases
