 ## SAM_AGI 
#### Super Autonomous (and Adaptive) Model 
#### Artificial General Intelligence

function context_aware_transfusion(input_data, models, contexts): 
    # Step 1: Get Input Data 
    InputData = ScreenShot or Video or Audio

    # Step 2: Infer Context
    context_probs = softmax([score_context(c, features) for c in contexts])
    context = argmax(context_probs)

    # Step 3: Compute Contextual Scaling Factors
    scaling_factors = softmax([score_submodel(i, context) for i in range(len(models))])

    # Step 4: Construct Transfusion Matrix
    transfusion_matrix = sum(
        scaling_factors[i] * concatenate(models[i]["weights"], models[i]["biases"])
        for i in range(len(models))
    )

    # Step 5: Apply Contextual Transformation
    transformed_matrix = apply_context_transformation(transfusion_matrix, context)

    # Step 6: Redistribute to Submodels
    for i in range(len(models)):
        models[i]["weights"], models[i]["biases"] = split(
            scaling_factors[i] * transformed_matrix
        )

    # Step 7: Optimize Submodels Locally
    for model in models:
        model["weights"] -= learning_rate * gradient * context (model["weights"])
        model["biases"] -= learning_rate * gradient * context (model["biases"])

    return models

## Mathematical Foundation

### Transfusion (Transformer → NEAT)

The transfusion process transfers knowledge from the Transformer model to the NEAT model using context-aware scaling:

T = ∑_{i=1}^k γᵢ(C(t)) \cdot P[i][j] \cdot \begin{bmatrix} W_t \\ B_t \end{bmatrix}

Where:
- T is the transfusion matrix
- γᵢ(C(t)) is the context-dependent scaling factor
- P[i][j] is the projection matrix
- [Wₜ, Bₜ] are the weights and biases of the transformer

The NEAT model then receives the transfused knowledge:

\begin{bmatrix} W_i' \\ B_i' \end{bmatrix} = γᵢ(C(t)) \cdot T

### Submodel Training

The NEAT model is trained using gradient descent with context-aware updates:

W_i^{(t+1)} = W_i' - η \cdot \frac{\partial L_i}{\partial W_i'}

B_i^{(t+1)} = B_i' - η \cdot \frac{\partial L_i}{\partial B_i'}

### Redistribution/Generalization (NEAT → Transformer)

Knowledge is redistributed back to the transformer using:

G = ∑_{i=1}^k βᵢ(Mᵢ, C(t)) \cdot \begin{bmatrix} W_i \\ B_i \end{bmatrix}

Where:
- G is the generalization matrix
- βᵢ(Mᵢ, C(t)) is the model and context-dependent scaling factor
- [Wᵢ, Bᵢ] are the weights and biases of the NEAT model

The transformer is then updated:

\begin{bmatrix} W_t^{(t+1)} \\ B_t^{(t+1)} \end{bmatrix} = W_t + λ \cdot G

## Implementation Details

### Context

The context C(t) is a single parameter that guides both transfusion and generalization:
- Values close to 0 indicate more reliance on the transformer
- Values close to 1 indicate more reliance on NEAT
- Context is updated based on model performance and task characteristics

### Knowledge Transfer

1. **Transfusion Process**:
   - Transformer weights are projected through context-aware scaling
   - NEAT receives scaled knowledge based on current context
   - Gradual transfer ensures stability

2. **Training**:
   - Both models train independently
   - Context influences learning rate and weight updates
   - Performance feedback adjusts context

3. **Generalization**:
   - NEAT knowledge is redistributed back to transformer
   - Context determines the strength of redistribution
   - Maintains balance between specialization and generalization

## Architecture

```
[Transformer] ←→ [Context] ←→ [NEAT]
     ↑                            ↑
     |                            |
  Project                    Generalize
     |                            |
     ↓                            ↓
[Input Data] ←→ [Task Space] ←→ [Output]
```

The system maintains a dynamic balance between the transformer's general knowledge and NEAT's specialized capabilities, guided by the context parameter.

## ONLINE AND OFFLINE Projection vs Generalization

### Initialization
- **Context Determination**: The first step in initializing the SAM model is determining the context, which is currently hardcoded but will later be derived using the ForeRunner model.
- **Transfusion Process**: Knowledge transfer occurs through a functional transfusion matrix, projecting weights and biases from the `head_model` (Transformer) to the `sub_model` (NEAT).
- **Completion**: Once the transfusion process is complete, the SAM model is set up and ready for runtime.

### Runtime
- **Forward and Backpropagation**: During runtime, the model performs forward and backward passes. If the answer is correct, the NEAT submodel grows by adding layers. If incorrect, backpropagation continues until the model achieves 100% accuracy.

### Post-Processing
- **Generalization**: After runtime, the weights from the `sub_model` are generalized back into the `head_model`. This process stores the knowledge in a generalized form, ready for future projections based on context.

### Projection Algorithm (In Development)
- The projection algorithm aims to divide each weight and bias by the context type, countering the multiplication during forward passes and gradient descent. This ensures that the model can adapt its knowledge projection to any situation or context.

## Model Hierarchy
- `Transformer_t HeadModel`: The primary model responsible for initial data processing.
- `NEAT_t SubModel`: The sub-model that adapts and learns from the transfused weights and biases.

## Initialization Process
- **Transfusion**: The first step involves the transfusion of weights and biases from the `HeadModel` to the `SubModel` via projection. This sets up the `SubModel` for adaptive learning.
- **Generalization**: After runtime, the context is used for the redistribution of weights and biases from the `SubModel` back into the generalized form of the `HeadModel`, effectively reversing the transfusion process.

Model's Architecture: 
    The model consists of a transformer head model that transfuses its weights to inner NEAT models through a transfusion matrix. The head model serves two purposes:
    1. Breaking down complex tasks into subtasks
    2. Transfusing learned knowledge to submodels

    The fitness functions are currently hardcoded:
    - Head model: Global fitness based on question-answering accuracy
    - NEAT submodels: Task-specific fitness functions optimized for their subtasks

    The hierarchical architecture can be summarized as:
    - Transformer head model
        - Transfusion matrices
            - NEAT parent models
                - Task-specific submodels

Model's Purpose:
    - The task at hand is to serve as a user input text to output text model which is different from a Large Language Model in the way that it does not need to be nearly as large as a Large Language Model. Which has an arbituary size and dimensions. Instead it has been built up with transformer/neat models that are less arbituary in the way that the model learned over time and built itself up and upon completion of tasks grows in size and dimensions. When the model grows in size and dimensions it is able to take on more complex tasks. 

Model's Training:
    - The model will be trained on text data and then tested on text data.

Model's Testing:
    - The model will be tested on text data.

Current Task:
    - Implement the transfusion process and the fitness function.

## Training on Text Data

### Dataset Requirements

- The dataset should be a text file located in the `datasets` folder.
- Each line in the file should represent a training sample, with input and target values separated by a delimiter (e.g., comma).

### Usage

To train the model on the text dataset, run the following command:

```bash
sh train.sh <dataset_path>
```
## Testing on Text Data

To test the model on the text dataset, run the following command:

```bash
sh test.sh <dataset_path>
```
