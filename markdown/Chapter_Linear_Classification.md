# Chapter 4: Linear Classification

## Learning Goals

By the end of this chapter, you will:

- **Know what is meant by binary linear classification** and understand its fundamental concepts
- **Understand why an explicit threshold for a classifier is redundant** and how bias terms can be eliminated using dummy features
- **Be able to specify weights and biases by hand** to represent simple logical functions (AND, OR, NOT)
- **Be familiar with input space and weight space**, including:
  - Plotting training cases and classification weights in both spaces
  - Understanding the geometric interpretation of linear classifiers
- **Be aware of the limitations of linear classifiers**, including:
  - Understanding convexity and its role in linear separability
  - Knowing how basis function representations can overcome some limitations

## Fundas: Mathematical Foundations

### Vector Representation
- **Input vectors**: Each data point is represented as a D-dimensional vector **x**^(i) = [x₁^(i), x₂^(i), ..., x_D^(i)]
- **Weight vectors**: Classification parameters represented as **w** = [w₁, w₂, ..., w_D]
- **Linear combination**: f(**x**) = **w**ᵀ**x** + b, where b is the bias term
- **Decision boundary**: The hyperplane where **w**ᵀ**x** + b = 0

### Binary Classification Framework
- **Target values**: t^(i) ∈ {0, 1} where 0 = negative class, 1 = positive class
- **Classification rule**: ŷ = 1 if **w**ᵀ**x** + b > τ, else ŷ = 0 (where τ is threshold)
- **Training set**: {(**x**^(i), t^(i))}ᵢ₌₁ᴺ where N is the number of examples

## Introduction to Binary Classification

Binary classification represents one of the most fundamental problems in machine learning, where the goal is to predict a binary-valued target from input features. This forms the foundation for understanding more complex classification scenarios.

### Real-World Applications

**Medical Diagnosis Systems**
- **Problem**: Predict whether a patient has a specific disease
- **Features**: Symptoms, test results, patient history
- **Target**: Disease present (1) or absent (0)
- **Example**: Diagnosing diabetes from glucose levels, BMI, and family history

**Email Spam Detection**
- **Problem**: Classify emails as spam or legitimate
- **Features**: Word frequencies, sender information, email metadata
- **Target**: Spam (1) or not spam (0)
- **Example**: Using keywords like "free money" as indicators

**Fraud Detection**
- **Problem**: Identify fraudulent transactions
- **Features**: Transaction amount, time, location, merchant type
- **Target**: Fraudulent (1) or legitimate (0)
- **Example**: Detecting unusual spending patterns

## Binary Linear Classifiers: First Principles

### Core Concept

A binary linear classifier makes decisions by computing a linear function of the input features and comparing the result to a threshold. This approach assumes that the two classes can be separated by a linear decision boundary in the feature space.

### Mathematical Formulation

**Step 1: Linear Combination**
```
z = w₁x₁ + w₂x₂ + ... + w_D x_D + b
```

**Step 2: Threshold Decision**
```
ŷ = 1 if z > τ, else ŷ = 0
```

Where:
- **w_i**: Weight for feature i (determines importance and direction)
- **x_i**: Value of feature i
- **b**: Bias term (shifts the decision boundary)
- **τ**: Threshold value

### Eliminating Redundancy: The Bias-Threshold Trick

**Problem**: Having both bias (b) and threshold (τ) is redundant.

**Solution**: Absorb threshold into bias term:
- Set new bias: b' = b - τ
- Set threshold to zero: τ = 0
- Decision rule becomes: ŷ = 1 if **w**ᵀ**x** + b' > 0

**Alternative Solution**: Add dummy feature:
- Extend input: **x** → [**x**, 1]
- Extend weights: **w** → [**w**, b]
- Decision rule: ŷ = 1 if **w**_extended ᵀ **x**_extended > 0

## Geometric Interpretation

### Input Space Perspective

**Decision Boundary**: The hyperplane **w**ᵀ**x** + b = 0 divides the input space into two regions:
- **Positive region**: **w**ᵀ**x** + b > 0 (predicted class 1)
- **Negative region**: **w**ᵀ**x** + b < 0 (predicted class 0)

**Properties**:
- The weight vector **w** is perpendicular to the decision boundary
- The bias b determines the distance of the boundary from the origin
- Moving along **w** increases the classifier's output

### Weight Space Perspective

**Constraint Regions**: Each training example imposes a constraint on the weight space:
- **Positive examples**: Require **w**ᵀ**x**^(i) + b > 0
- **Negative examples**: Require **w**ᵀ**x**^(i) + b < 0

**Feasible Region**: The intersection of all constraints defines the region of weight vectors that correctly classify all training examples.

## Hand-Designed Logic Functions

### AND Function
**Truth Table**: (0,0)→0, (0,1)→0, (1,0)→0, (1,1)→1

**Implementation**:
- Weights: w₁ = 1, w₂ = 1
- Bias: b = -1.5
- Decision: ŷ = 1 if x₁ + x₂ - 1.5 > 0

**Verification**:
- (0,0): 0 + 0 - 1.5 = -1.5 < 0 → ŷ = 0 ✓
- (1,1): 1 + 1 - 1.5 = 0.5 > 0 → ŷ = 1 ✓

### OR Function
**Implementation**:
- Weights: w₁ = 1, w₂ = 1
- Bias: b = -0.5
- Decision: ŷ = 1 if x₁ + x₂ - 0.5 > 0

### NOT Function
**Implementation**:
- Weight: w₁ = -1
- Bias: b = 0.5
- Decision: ŷ = 1 if -x₁ + 0.5 > 0

## Limitations of Linear Classifiers

### Linear Separability

**Definition**: A dataset is linearly separable if there exists a linear decision boundary that perfectly separates the two classes.

**Convexity and Separability**:
- **Convex sets**: A set where the line segment between any two points lies entirely within the set
- **Key insight**: If the positive examples form a convex set and the negative examples form a convex set, and these sets don't overlap, then the data is linearly separable

### The XOR Problem

**XOR Truth Table**: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0

**Why XOR is not linearly separable**:
- Positive examples: (0,1) and (1,0)
- Negative examples: (0,0) and (1,1)
- No single line can separate these points correctly

### Overcoming Limitations: Basis Functions

**Strategy**: Transform inputs using basis functions φ(**x**) to create a new feature space where linear separation becomes possible.

**Example for XOR**:
- Original features: x₁, x₂
- Basis functions: φ₁ = x₁, φ₂ = x₂, φ₃ = x₁x₂
- In the new space [x₁, x₂, x₁x₂], XOR becomes linearly separable

## Practical Considerations

### Feature Engineering
- **Normalization**: Scale features to similar ranges
- **Interaction terms**: Add products of features (x₁x₂)
- **Polynomial features**: Add powers of features (x₁², x₂²)

### Model Selection
- **Bias-variance tradeoff**: Simple models (few features) vs. complex models (many features)
- **Interpretability**: Linear classifiers provide clear feature importance through weights
- **Computational efficiency**: Linear models are fast to train and evaluate

### Performance Evaluation
- **Training accuracy**: Percentage of training examples correctly classified
- **Generalization**: Performance on unseen test data
- **Decision boundary visualization**: Plot boundaries in 2D for intuition

## Chapter Summary

Linear classification provides a fundamental framework for understanding binary classification problems. Key takeaways:

1. **Geometric intuition**: Linear classifiers create hyperplane decision boundaries in input space
2. **Mathematical elegance**: Simple linear combinations with threshold decisions
3. **Practical utility**: Many real-world problems are approximately linearly separable
4. **Clear limitations**: Not all problems (like XOR) can be solved with linear classifiers
5. **Extension possibilities**: Basis functions can transform non-linearly separable problems

The concepts developed here—particularly the geometric interpretation in both input and weight spaces—form the foundation for understanding more sophisticated classification algorithms like the perceptron, support vector machines, and neural networks.

## Next Steps

The next chapter will introduce the **perceptron algorithm**, which provides a systematic method for learning the weights of a linear classifier from training data, moving beyond hand-designed solutions to automated learning procedures.