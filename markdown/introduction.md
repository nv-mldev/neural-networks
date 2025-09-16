# Introduction to Machine Learning

## What is Learning?

Learning is the process of acquiring new knowledge, skills, or behaviors through experience. This process transforms inputs—such as data, experiences, or information—into useful capabilities like expertise, new skills, or predictive models.

### Key Questions in Learning

- What are the essential inputs for the learning process?
- How do we measure the effectiveness and success of learning?
- What are the underlying mechanisms and processes by which learning occurs?

## What is Reasoning?

Reasoning is the ability to draw logical conclusions from known facts or learned knowledge. Unlike learning, reasoning relies on logical inference rather than large amounts of data.

## From Animal Learning to Machine Learning

### Example: Bait Shyness in Rats

Rats demonstrate a fundamental learning principle through their feeding behavior:

- They sample novel food cautiously
- If the food causes illness, they avoid it in the future
- Past experience directly informs future decisions

This natural learning process parallels challenges in machine learning.

### Parallel: Spam Email Filtering

Consider how this biological learning principle applies to spam detection:

- **Naive approach**: Memorize all past spam emails
- **Problem**: Cannot classify previously unseen emails
- **Solution**: Extract generalizable patterns (like words, phrases, or sender patterns)
- **Key insight**: Both rats and spam filters must generalize from specific experiences to handle new, similar situations

This ability to generalize leads us to examine the different types of reasoning that enable learning and decision-making.

## Types of Reasoning: Comprehensive Overview

### 1. Inductive Reasoning

**Definition:** Inductive reasoning extracts patterns from observed data to make predictions about future or unseen cases. This approach moves from specific observations to general conclusions, yielding probable rather than certain results.

**Key Characteristics:**

- Most prevalent form of reasoning in the animal kingdom and primary mode in machine learning
- Forms the basis of most learned behaviors in animals  
- Used extensively in deep learning and LLMs

#### Examples Across Different Contexts

**Human Example**: Every cat I have ever seen has four legs. Therefore, all cats have four legs.

**Animal Example**:

- A dog learns that when its owner picks up the leash, it will probably go for a walk (experienced hundreds of times)
- A squirrel learns that acorns are edible after eating many without getting sick

**Machine Learning Example**: A spam classifier learns from previously labeled emails and generalizes patterns to detect new spam messages

**LLM Example**: When asked to complete "The sky is blue because...", the model has "observed" this pattern countless times in training data and induces probable completions based on statistical patterns

#### Applications in AI/ML

- Deep learning model training
- Pattern recognition systems
- Predictive analytics
- Natural language processing

### 2. Deductive Reasoning

**Definition:** Deductive reasoning moves from general rules and premises to reach specific, guaranteed conclusions. It starts with a general rule and a specific case to reach a logical conclusion.

**Key Characteristics:**

- Provides certainty when premises are true
- Animals generally lack this capability for abstract reasoning
- LLMs can only mimic this through pattern matching

#### Examples Across Different Contexts

**Human Example**: All men are mortal. Socrates is a man. Therefore, Socrates is mortal.

**Mathematical Example**: If all squares have four sides and a shape is a square, it must have four sides.

**Animal Limitation**: Animals cannot perform abstract syllogistic reasoning, such as deducing that "because all felines are carnivores and a tiger is a feline, then a tiger is a carnivore."

**LLM Mimicry**: When given "All mammals have hair. A dolphin is a mammal. Therefore...", the model completes with "a dolphin has hair"—not through true logical syllogism, but by recognizing learned textual patterns from training data.

#### Applications in AI

- Expert systems (traditional AI)
- Symbolic reasoning systems
- Theorem proving
- Rule-based systems

#### Limitations

- LLMs lack strict logical reasoning capabilities
- Most modern AI systems don't use true deductive reasoning

### 3. Abductive Reasoning (Inference to Best Explanation)

**Definition:** Abductive reasoning starts with an observation and seeks to find the simplest and most likely explanation. It's the process of finding a hypothesis that, if true, would best explain the observation.

#### Key Characteristics

- Often described as "inference to the best explanation"
- Guesses the most probable explanation given incomplete data
- Can be demonstrated in simple forms by animals
- Simulated effectively by modern LLMs

#### Examples Across Different Contexts

**Human Example**: The grass is wet. A plausible explanation is that it rained (most likely, though sprinklers are possible).

**Medical Example**: A doctor observes symptoms like fever and cough and infers the patient likely has the flu.

**Animal Example**:

- A squirrel hears rustling and sees movement, "abduces" it's a predator and climbs a tree
- A raven sees a human place a rock over food, infers the food is under the rock when human leaves

**LLM Example**: When asked "Why is the road wet?", generates explanations like "It rained," "Water main broke," or "Street cleaner passed" by ranking probable explanations from training data.

#### Applications in AI/ML

- Medical diagnosis systems
- Troubleshooting AI
- Natural language understanding
- Creative writing and content generation

#### In Machine Learning Systems

**Hypothesis Generation**: ML models generate predictions based on learned patterns - they present the most probable explanation without "knowing" it's true.

**Bayesian Inference**: AI systems use Bayesian models to start with prior beliefs and update them as new evidence is presented.

### 4. Analogical Reasoning (Pattern Transfer)

**Definition:** Analogical reasoning applies knowledge from one context to another by recognizing similar patterns or relationships.

#### Applications in AI/ML

- AI-powered tutoring systems
- Cross-domain learning
- Transfer learning in neural networks

#### Example

AI that learns human speech patterns in English and transfers that learning to generate speech in another language.

### 5. Bayesian Reasoning (Probabilistic Prediction)

**Definition:** Bayesian reasoning uses probability to predict outcomes by updating beliefs based on new evidence.

#### Applications in AI/ML

- Spam filtering systems
- AI language models
- Uncertainty quantification

#### Example

A Bayesian spam filter assigns probabilities to words appearing in spam emails and calculates the likelihood that a new email is spam.

### 6. Causal Reasoning (Understanding Cause-and-Effect)

**Definition:** Causal reasoning determines causal relationships rather than just correlations.

#### Limitations in Current AI

- LLMs struggle with true causality
- Most AI systems identify correlations rather than causes

#### Example

In healthcare, researchers identify that smoking causes lung cancer, rather than just observing that smokers have higher cancer rates.

### 7. Counterfactual Reasoning (What-If Thinking)

**Definition:** Counterfactual reasoning explores hypothetical scenarios and alternative possibilities.

#### Applications in AI

- Risk analysis systems
- AI decision-making
- Simulation and planning

#### Example

A self-driving car AI simulates different driving scenarios to decide the safest course of action in an emergency.

## Reasoning Capabilities Across Intelligence Types

| **Reasoning Type** | **Animals** | **Large Language Models** | **Traditional AI** |
|--------------------|-------------|---------------------------|--------------------|
| **Inductive** | Primary (survival-focused) | Primary (pattern-based) | Limited |
| **Deductive** | Absent (complex forms) | Simulated (pattern matching) | Primary (rule-based) |
| **Abductive** | Limited (simple forms) | Effective (learned patterns) | Limited |
| **Causal** | Basic | Limited | Rule-dependent |
| **Adaptability** | High (within domain) | High (pattern recognition) | Low (manual updates) |

While inductive reasoning is powerful, it has inherent limitations that both animals and AI systems must navigate.

## Limitations of Inductive Reasoning

### Pigeon Superstition Experiment (B.F. Skinner)

This experiment demonstrated how animals can form false associations through inductive reasoning.

### Garcia & Koelling Experiment (1966)

This landmark experiment studied **selective associative learning** in rats and demonstrated that **not all stimuli are equally associated** with consequences.

**Experimental Design:**

Researchers used a compound stimulus approach:

- **Taste component**: Saccharin-flavored water
- **Audiovisual component**: Lights and sounds during drinking

Rats were then exposed to different aversive consequences:

- **Group 1**: Illness (nausea from mild radiation or toxin)
- **Group 2**: Physical discomfort (mild electric shocks)

**Results:**

**Illness-Induced Group:**

- Developed strong aversion to taste cues (saccharin water)
- Showed minimal aversion to audiovisual cues

**Shock-Induced Group:**

- Developed strong aversion to audiovisual cues (lights and sounds)  
- Showed no aversion to taste cues

**Key Finding:** Rats selectively associated specific stimuli with appropriate consequences—taste with illness, external cues with physical danger.

**Scientific Impact:**

This experiment revolutionized learning theory by:

- **Challenging equipotentiality**: Not all stimulus-response associations are equally learnable
- **Demonstrating biological constraints**: Evolution shapes what animals can easily learn
- **Revealing adaptive biases**: Learning mechanisms evolved to enhance survival
  - Taste naturally links to internal consequences (poisoning)
  - External cues (sounds, lights) link to external threats (predators)

**Implications for Machine Learning:**

- **Learning requires inductive bias**: Not all associations are equally learnable
- **Feature relevance varies**: Some inputs are more informative than others  
- **Domain knowledge matters**: Evolutionary or expert-designed constraints improve learning
- **No universal learner exists**: All learning algorithms must make assumptions (No-Free-Lunch theorem)

These biological insights directly inform machine learning design, where inductive bias plays a crucial role in model performance.

## Inductive Bias in Machine Learning

### What is Inductive Bias?

**Definition:** Inductive bias refers to the set of assumptions that a learning algorithm makes to generalize from limited training data to unseen data.

**Why is it Critical?**

Inductive bias is essential because:

- Machine learning models have limited training data
- Models must generalize from past observations to unseen cases  
- Without appropriate bias, models may overfit (memorizing training data without learning generalizable patterns)

### Types of Inductive Biases

#### 1. Preference for Simpler Models (Occam's Razor)

- **Assumption**: Simpler explanations are preferred over complex ones
- **Example**: Decision trees with fewer splits are preferred because they generalize better
- **In Deep Learning**: Regularization techniques (L1, L2) penalize complex models

#### 2. Smoothness Assumption

- **Assumption**: Data points that are close together should have similar outputs
- **Example**: In image classification, two similar images should belong to the same class
- **In ML**: K-Nearest Neighbors (KNN) assumes nearby data points have the same label

#### 3. Similar Features Should Have Similar Effects

- **Assumption**: If two features are related, their effects should be similar
- **Example**: In linear regression, correlated features often have similar coefficients

#### 4. Prior Knowledge About the Task (Domain-Specific Bias)

- **Assumption**: Certain relationships are more likely in specific tasks
- **Example**: In NLP, word order matters
- **In ML**: Transformers use positional embeddings to capture sentence structure

#### 5. Invariance Bias (Translation, Rotation, Scale Invariance)

- **Assumption**: Some transformations should not change predictions
- **Example**: Rotating an image of a cat should still classify it as a cat
- **In ML**: CNNs use convolutional filters to enforce translation invariance

#### 6. Sparsity Assumption

- **Assumption**: Only a few features are truly important
- **Example**: In text classification, most words are irrelevant
- **In ML**: L1 regularization forces models to select important features

These general principles manifest differently across various neural network architectures, each designed with specific inductive biases for particular tasks.

## Inductive Bias in Specific Architectures

### Convolutional Neural Networks (CNNs)

CNNs are designed for image processing and rely on three key inductive biases:

#### 1. Locality Bias (Local Connectivity)

- **Assumption**: Nearby pixels are more relevant than distant pixels
- **Example**: In facial recognition, CNN detects eyes, nose, mouth before recognizing entire face

#### 2. Translation Invariance

- **Assumption**: An object should be recognized regardless of position
- **How it works**: CNNs use shared convolutional filters
- **Example**: Handwritten digit "3" recognized anywhere in the image

#### 3. Hierarchical Feature Learning

- **Assumption**: Complex patterns learned by stacking abstraction layers
- **Example**: Lower layers detect edges → middle layers detect shapes → deeper layers detect objects

### Recurrent Neural Networks (RNNs & LSTMs)

RNNs are designed for sequential data and rely on:

#### 1. Temporal Dependency Bias

- **Assumption**: Recent information is more important than distant past
- **Example**: In "The cat sat on the mat", nearby words are more related

#### 2. Order Sensitivity Bias

- **Assumption**: The order of input elements matters
- **Example**: "Dog bites man" ≠ "Man bites dog"

### Transformers (BERT, GPT)

#### 1. Attention-Based Bias (Self-Attention)

- **Assumption**: Important words can be anywhere in a sentence
- **Example**: In "The dog chased the ball...which was blue", "which" refers to "ball"

#### 2. Context-Aware Learning Bias

- **Assumption**: Word meaning depends on context
- **Example**: "Bank" can mean financial institution or riverbank

#### 3. Positional Encoding Bias

- **Assumption**: Order matters even without sequential processing
- **Example**: "She ate an apple" ≠ "An apple ate she"

## Machine Learning vs Other Approaches

### Why Machine Learning?

- For many problems, it's difficult to program correct behavior by hand
- Examples: recognizing objects in images, understanding human speech
- Machine learning approach: program an algorithm to automatically learn from data

### Reasons to Use Learning Algorithms

- Hard to code up a solution by hand (e.g., vision, speech)
- System needs to adapt to changing environment (e.g., spam detection)
- Want system to perform better than human programmers
- Privacy/fairness considerations (e.g., ranking search results)

### Machine Learning vs Statistics

| **Aspect** | **Statistics** | **Machine Learning** |
|------------|----------------|----------------------|
| **Primary Goal** | Draw valid conclusions for scientists/policymakers | Build autonomous predictive systems |
| **Focus** | Interpretability and mathematical rigor | Predictive performance and scalability |
| **Approach** | Hypothesis testing and inference | Pattern recognition and automation |

### Relations to AI

- AI does not always imply a learning-based system
- **Non-learning AI approaches**:
  - Symbolic reasoning
  - Rule-based systems
  - Tree search
- **Learning-based systems**:
  - Learned based on data
  - More flexibility
  - Good at solving pattern recognition problems

## Symbolic AI vs Machine Learning

### What is Symbolic AI?

- Also known as **Good Old-Fashioned AI (GOFAI)**
- Represents knowledge using **symbols, rules, and logic**
- Uses **explicitly programmed rules** for reasoning
- Based on **formal logic, tree search, and knowledge representation**
- **Example**: If "All humans are mortal" and "Socrates is human", then "Socrates is mortal"

### Tree Search in Symbolic AI

- Fundamental approach for solving problems by exploring decision trees
- Uses algorithms like DFS, BFS, and A* Search
- **Example**: Chess AI searches possible future board states
- Does **not generalize from data** but computes solutions using logical steps

### Rule-Based AI

- Subset of Symbolic AI using **explicit if-then rules**
- Rules manually defined by experts
- **Examples**:
  - IF patient has fever AND cough → Diagnose as flu
  - IF transaction > $10,000 → Flag as potential fraud
- **Limitation**: Struggles with exceptions and uncertain scenarios

### Symbolic AI vs Machine Learning Comparison

| **Feature** | **Symbolic AI** | **Machine Learning** |
|-------------|-----------------|----------------------|
| **Knowledge Source** | Rules & logic | Data & patterns |
| **Interpretability** | Highly explainable | Often a black box |
| **Adaptability** | Rigid (manual updates) | Can generalize from data |
| **Data Requirements** | Minimal | Requires large datasets |
| **Best Use Cases** | Theorem proving, expert systems | NLP, computer vision, recommendations |

### Limitations of Symbolic AI

- Difficult to scale—requires manual updates
- Cannot handle unstructured data (images, speech)
- Struggles with uncertainty—rules don't adapt
- Machine Learning outperforms in perception tasks

### The Future: Hybrid AI (Neuro-Symbolic AI)

- Combines **Symbolic AI (structured logic)** with **Machine Learning (pattern recognition)**
- **Example**: AI lawyer uses Symbolic AI for legal rules + ML for case history analysis
- Enhances **explainability, adaptability, and reasoning**

## Key Takeaways

### Fundamental Principles

- **Learning transforms experience into generalizable knowledge** through pattern recognition and abstraction
- **Inductive bias is essential** for effective learning—all successful learning systems require appropriate assumptions about their domain
- **No universal learner exists**—every learning algorithm must make trade-offs based on its intended use case

### Reasoning in AI Systems

- **Biological systems** primarily use inductive reasoning, with limited abductive capabilities
- **Traditional AI systems** excel at deductive reasoning but struggle with adaptation
- **Modern ML/LLMs** are powerful at inductive reasoning and can simulate other reasoning types through pattern matching
- **Future AI systems** will likely combine symbolic reasoning with machine learning for enhanced interpretability and robust decision-making

### Practical Implications

- **Domain knowledge matters**—incorporating appropriate inductive biases improves learning efficiency and generalization
- **Architecture design is crucial**—different neural network architectures embed specific assumptions about data structure (CNNs for images, Transformers for sequences)
- **Learning and reasoning are complementary**—effective AI systems need both the ability to learn from data and to reason with acquired knowledge

## Role of Prior Knowledge

The effectiveness of any learning system—biological or artificial—depends heavily on the prior knowledge and assumptions it brings to new situations.

**Key Principles:**

- **Prior knowledge accelerates learning**: Domain expertise makes learning from examples more efficient
- **Trade-offs exist**: Stronger assumptions improve performance on expected tasks but reduce flexibility for unexpected scenarios  
- **Balance is crucial**: The optimal learning system finds the right equilibrium between prior knowledge and data-driven adaptation
- **Tool development matters**: Creating methods to incorporate domain expertise remains central to advancing machine learning theory
