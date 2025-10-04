An important geometric concept which helps us out here is convexity. A
set S is convex if the line segment connecting any two points in S must lie
within S. It’s not too hard to show that if S is convex, then any weighted
average of points in S must also lie within S. A weighted average of points
x(1)
,...,x(N ) is a point given by the linear combination
x(avg) = λ1x(1) +···+ λN x(N )
,
where 0 ≤λi ≤1 and λ1 +···+ λN = 1. You can think of the weighted
average as the center of mass, where the mass of each point is given by λi.

The weighted average of a set of points will always lie within the convex hull of those points. The convex hull is the smallest convex shape that contains all the points

In the context of binary classification, there are two important sets that
are always convex:

1. In data space, the positive and negative regions are both convex. Both
regions are half-spaces, and it should be visually obvious that half-
spaces are convex. This implies that if inputs x(1)
,...,x(N ) are all
in the positive region, then any weighted average must also be in the
positive region. Similarly for the negative region.

2. In weight space, the feasible region is convex. The rough mathematical
argument is as follows. Each good region (the set of weights which
correctly classify one data point) is convex because it’s a half-space.
The feasible region is the intersection of all the good regions, so it
must be convex because the intersection of convex sets is convex.

### Example

Let’s return to the XOR example. Since the posi-
tive region is convex, if we draw the line segment connecting the
two positive examples (0,1) and (1,0), this entire line segment
must be classified as positive. Similarly, if we draw the line seg-
ment connecting the two negative examples (0,0) and (1,1), the
entire line segment must be classified as negative. But these two
line segments intersect at (0.5,0.5), which means this point must
be classified as both positive and negative, which is impossible.
(See Figure 3.) Therefore, XOR isn’t linearly separable.

![alt text](figures/pattern.png)
Our last example was somewhat artificial. Let’s
now turn to a somewhat more troubling, and practically relevant,
limitation of linear classifiers. Let’s say we want to give a robot
a vision system which can recognize objects in the world. Since
the robot could be looking any given direction, it needs to be
able to recognize objects regardless of their location in its visual
field. I.e., it should be able to recognize a pattern in any possible
translation.
As a simplification of this situation, let’s say our inputs are
16-dimensional binary vectors and we want to distinguish two
patterns, A, and B (shown in Figure 4), which can be placed in
any possible translation, with wrap-around. (I.e., if you shift the
pattern right, then whatever falls oﬀ the right side reappears on
the left.) Thus, there are 16 examples of A and 16 examples of
B that our classifier needs to distinguish.
By convexity, if our classifier is to correctly classify all 16 in-
stances of A, then it must also classify the average of all 16. instances as A. Since 4 out of the 16 values are on, the aver-
age of all instances is simply the vectors (0.25,0.25,...,0.25).
Similarly, for it to correctly classify all 16 instances of B, it
must also classify their average as B. But the average is also
(0.25,0.25,...,0.25). Since this vector can’t possibly be classi-
fied as both A and B, this dataset must not be linearly separable.
More generally, we can’t expect any linear classifier to detect a
pattern in all possible translations. This is a serious limitation
of linear classifiers as a basis for a vision system.

Circumventing this problem by using feature represen-
tations
We just saw a negative result about linear classifiers. Let’s end on a more
positive note. In Lecture 2, we saw how linear regression could be made
more powerful using a basis function, or feature, representation. The same
trick applies to classification. Essentially, in place of z = w T x + b, we use
z = wT φ(x) + b, where φ(x) = (φ1(x), . . . , φD (x)) is a function mapping
input vectors to feature vectors. Let’s see how we can represent XOR using
carefully selected features.
Example 5. Consider the following feature representation for
XOR:
φ1 (x) = x1
φ2 (x) = x2
φ3 (x) = x1 x2
In this representation, our training set becomes
φ1 (x)
 φ2(x)
 φ3 (x)
 t
0
 0
 0
 0
0
 1
 0
 1
1
 0
 0
 1
1
 1
 1
 0
Using the same techniques as in Examples 1 and 2, we find that
the following set of weights and biases correctly classifies all the
training examples:
b = −0.5
 w1 = 1
 w2 = 1
 w3 = −2.
The only problem is, where do we get the features from? In this example,
we just pulled them out of a hat. Unfortunately, there’s no recipe for coming
up with good features, which is part of what makes machine learning hard.
But next week, we’ll see how we can learn a set of features by training a
multilayer neural net.
