#ifndef HF_INTERFACE_H
#define HF_INTERFACE_H

#include <stdlib.h>
#include <stddef.h>

// Hugging Face model handle
typedef void* HF_Model;

// Initialize a Hugging Face model
// model_name: e.g., "gpt2", "bert-base-uncased", "distilbert-base-uncased"
// Returns: Model handle or NULL on error
HF_Model HF_init_model(const char* model_name);

// Free a Hugging Face model
void HF_free_model(HF_Model model);

// Get embeddings from HF model
// model: Model handle
// text: Input text
// embeddings: Output array (must be pre-allocated, size = model_dim)
// model_dim: Dimension of the model's embeddings
// Returns: 1 on success, 0 on error
int HF_get_embeddings(HF_Model model, const char* text, long double* embeddings, size_t model_dim);

// Generate text using HF model
// model: Model handle
// prompt: Input prompt
// max_length: Maximum generation length
// output: Output buffer (must be pre-allocated)
// output_size: Size of output buffer
// Returns: 1 on success, 0 on error
int HF_generate_text(HF_Model model, const char* prompt, size_t max_length, char* output, size_t output_size);

// Get model dimension
// model: Model handle
// Returns: Model dimension or 0 on error
size_t HF_get_model_dim(HF_Model model);

// Train SAM using HF model as teacher
// hf_model: Hugging Face model (teacher)
// sam_model: SAM model (student)
// training_data: Array of training texts
// num_samples: Number of training samples
// epochs: Number of training epochs
// Returns: 1 on success, 0 on error
int HF_train_sam(HF_Model hf_model, void* sam_model, const char** training_data, size_t num_samples, size_t epochs);

#endif // HF_INTERFACE_H

