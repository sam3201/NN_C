#ifndef SNN_H
#define SNN_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Spiking Neural Network Neuron structure
typedef struct SNNNeuron {
  long double membrane_potential;
  long double threshold;
  long double reset_potential;
  long double refractory_period;
  int refractory_counter;
  long double *input_weights;
  size_t num_inputs;
  long double last_spike_time;
  long double tau_mem;  // Membrane time constant
  long double tau_syn;  // Synaptic time constant
  long double *synaptic_currents;
  bool has_spiked;
} SNNNeuron;

// Spiking Neural Network structure
typedef struct SNN {
  size_t num_neurons;
  size_t input_size;
  size_t output_size;
  SNNNeuron *neurons;
  long double **spike_trains;
  long double simulation_time;
  long double dt;  // Time step
  long double current_time;
  size_t max_spike_trains_length;
  bool use_current_based;  // Current-based vs voltage-based
} SNN_t;

// Spiking Neural Network functions
SNN_t *SNN_create(size_t num_neurons, size_t input_size, size_t output_size);
void SNN_destroy(SNN_t *snn);
long double *SNN_forward(SNN_t *snn, long double *inputs, long double duration);
void SNN_reset(SNN_t *snn);
void SNN_set_neuron_threshold(SNN_t *snn, size_t neuron_id, long double threshold);
void SNN_set_time_constant(SNN_t *snn, size_t neuron_id, long double tau_mem);
void SNN_set_time_step(SNN_t *snn, long double dt);
void SNN_add_synapse(SNN_t *snn, size_t from_neuron, size_t to_neuron, long double weight);
void SNN_remove_synapse(SNN_t *snn, size_t from_neuron, size_t to_neuron);
long double SNN_get_firing_rate(SNN_t *snn, size_t neuron_id, long double window);
long double *SNN_get_spike_times(SNN_t *snn, size_t neuron_id);
void SNN_set_current_based(SNN_t *snn, bool use_current_based);
size_t SNN_count_spikes(SNN_t *snn, long double time_window);
void SNN_print_summary(SNN_t *snn);

// Spike-based learning
void SNN_stdp_update(SNN_t *snn, size_t pre_neuron, size_t post_neuron, 
                    long double pre_spike_time, long double post_spike_time,
                    long double A_plus, long double A_minus, long double tau_plus, long double tau_minus);
void SNN_set_stdp_params(SNN_t *snn, long double A_plus, long double A_minus, 
                        long double tau_plus, long double tau_minus);

#ifdef __cplusplus
}
#endif

#endif // SNN_H
