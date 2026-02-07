#include "SNN.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Helper functions
static long double exponential_decay(long double current, long double tau, long double dt) {
    return current * expl(-dt / tau);
}

// Spiking Neural Network creation
SNN_t *SNN_create(size_t num_neurons, size_t input_size, size_t output_size) {
    SNN_t *snn = malloc(sizeof(SNN_t));
    if (!snn) return NULL;
    
    snn->num_neurons = num_neurons;
    snn->input_size = input_size;
    snn->output_size = output_size;
    snn->simulation_time = 0.0L;
    snn->dt = 0.1L;  // 10ms time step
    snn->current_time = 0.0L;
    snn->max_spike_trains_length = 1000;
    snn->use_current_based = false;
    
    // Create neurons
    snn->neurons = malloc(num_neurons * sizeof(SNNNeuron));
    if (!snn->neurons) {
        free(snn);
        return NULL;
    }
    
    // Initialize neurons
    for (size_t i = 0; i < num_neurons; i++) {
        SNNNeuron *neuron = &snn->neurons[i];
        
        neuron->membrane_potential = 0.0L;
        neuron->threshold = 1.0L;
        neuron->reset_potential = 0.0L;
        neuron->refractory_period = 2.0L;
        neuron->refractory_counter = 0;
        neuron->num_inputs = input_size;
        neuron->last_spike_time = 0.0L;
        neuron->tau_mem = 20.0L;  // 20ms membrane time constant
        neuron->tau_syn = 5.0L;   // 5ms synaptic time constant
        neuron->has_spiked = false;
        
        // Initialize input weights
        neuron->input_weights = calloc(input_size, sizeof(long double));
        neuron->synaptic_currents = calloc(input_size, sizeof(long double));
        
        // Initialize weights with normal distribution
        for (size_t j = 0; j < input_size; j++) {
            neuron->input_weights[j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.1L;
        }
    }
    
    // Create spike trains
    snn->spike_trains = malloc(num_neurons * sizeof(long double*));
    if (!snn->spike_trains) {
        free(snn->neurons);
        free(snn);
        return NULL;
    }
    
    for (size_t i = 0; i < num_neurons; i++) {
        snn->spike_trains[i] = malloc(snn->max_spike_trains_length * sizeof(long double));
        if (!snn->spike_trains[i]) {
            free(snn->neurons[i]);
            free(snn->spike_trains);
            free(snn);
            return NULL;
        }
        
        // Initialize spike trains with no spikes
        for (size_t j = 0; j < snn->max_spike_trains_length; j++) {
            snn->spike_trains[i][j] = -1.0L;  // -1 indicates no spike
        }
    }
    
    return snn;
}

// Spiking Neural Network destruction
void SNN_destroy(SNN_t *snn) {
    if (!snn) return;
    
    // Free neurons
    for (size_t i = 0; i < snn->num_neurons; i++) {
        free(snn->neurons[i].input_weights);
        free(snn->neurons[i].synaptic_currents);
        free(snn->spike_trains[i]);
    }
    free(snn->neurons);
    free(snn->spike_trains);
    free(snn);
}

// Set neuron threshold
void SNN_set_neuron_threshold(SNN_t *snn, size_t neuron_id, long double threshold) {
    if (snn && neuron_id < snn->num_neurons) {
        snn->neurons[neuron_id].threshold = threshold;
    }
}

// Set time constant
void SNN_set_time_constant(SNN_t *snn, size_t neuron_id, long double tau_mem) {
    if (snn && neuron_id < snn->num_neurons) {
        snn->neurons[neuron_id].tau_mem = tau_mem;
    }
}

// Set time step
void SNN_set_time_step(SNN_t *snn, long double dt) {
    if (snn) {
        snn->dt = dt;
    }
}

// Add synapse between neurons
void SNN_add_synapse(SNN_t *snn, size_t from_neuron, size_t to_neuron, long double weight) {
    if (!snn || from_neuron >= snn->num_neurons || to_neuron >= snn->num_neurons) return;
    
    // For output neurons, we don't have input weights
    if (to_neuron >= snn->num_neurons - snn->output_size) return;
    
    SNNNeuron *neuron = &snn->neurons[to_neuron];
    if (from_neuron < snn->input_size) {
        neuron->input_weights[from_neuron] = weight;
    }
}

// Remove synapse between neurons
void SNN_remove_synapse(SNN_t *snn, size_t from_neuron, size_t to_neuron) {
    if (!snn || from_neuron >= snn->num_neurons || to_neuron >= snn->num_neurons) return;
    
    // For output neurons, we don't have input weights
    if (to_neuron >= snn->num_neurons - snn->output_size) return;
    
    SNNNeuron *neuron = &snn->neurons[to_neuron];
    if (from_neuron < snn->input_size) {
        neuron->input_weights[from_neuron] = 0.0L;
    }
}

// Process input through SNN
long double *SNN_forward(SNN_t *snn, long double *inputs, long double duration) {
    if (!snn || !inputs) return NULL;
    
    long double end_time = snn->current_time + duration;
    
    // Clear spike trains
    for (size_t i = 0; i < snn->num_neurons; i++) {
        snn->neurons[i].has_spiked = false;
    }
    
    // Simulate for the specified duration
    while (snn->current_time < end_time) {
        // Update synaptic currents
        for (size_t i = 0; i < snn->num_neurons; i++) {
            SNNNeuron *neuron = &snn->neurons[i];
            
            if (neuron->refractory_counter > 0) {
                neuron->refractory_counter--;
                continue;
            }
            
            // Calculate synaptic current
            long double synaptic_current = 0.0L;
            for (size_t j = 0; j < neuron->num_inputs; j++) {
                if (j < snn->input_size && inputs[j] != 0.0L) {
                    synaptic_current += inputs[j] * neuron->input_weights[j];
                }
            }
            neuron->synaptic_currents[j] = exponential_decay(neuron->synaptic_currents[j], neuron->tau_syn, snn->dt);
            
            // Update membrane potential
            long double dV = (-neuron->membrane_potential + synaptic_current + neuron->rest_potential) / neuron->tau_mem;
            neuron->membrane_potential += dV * snn->dt;
            
            // Check for spike
            if (neuron->membrane_potential >= neuron->threshold && !neuron->has_spiked) {
                neuron->has_spiked = true;
                neuron->last_spike_time = snn->current_time;
                neuron->refractory_counter = (int)(neuron->refractory_period / snn->dt);
                neuron->membrane_potential = neuron->reset_potential;
                
                // Record spike
                size_t spike_index = (size_t)(snn->current_time / snn->dt);
                if (spike_index < snn->max_spike_trains_length) {
                    snn->spike_trains[i][spike_index] = snn->current_time;
                }
            }
        }
        
        snn->current_time += snn->dt;
    }
    
    // Collect output from output neurons
    long double *output = calloc(snn->output_size, sizeof(long double));
    if (!output) return NULL;
    
    for (size_t i = 0; i < snn->output_size; i++) {
        SNNNeuron *neuron = &snn->neurons[snn->num_neurons - snn->output_size + i];
        
        // Output is firing rate over the simulation period
        long double firing_rate = 0.0L;
        for (size_t j = 0; j < snn->max_spike_trains_length; j++) {
            if (snn->spike_trains[neuron->num_neurons - snn->output_size + i][j] >= 0.0L && 
                snn->spike_trains[neuron->num_neurons - snn->output_size + i][j] <= end_time) {
                firing_rate += 1.0L;
            }
        }
        output[i] = firing_rate / (end_time / snn->dt);
    }
    
    return output;
}

// Reset network state
void SNN_reset(SNN_t *snn) {
    if (!snn) return;
    
    snn->current_time = 0.0L;
    
    for (size_t i = 0; i < snn->num_neurons; i++) {
        SNNNeuron *neuron = &snn->neurons[i];
        neuron->membrane_potential = neuron->reset_potential;
        neuron->refractory_counter = 0;
        neuron->has_spiked = false;
        neuron->last_spike_time = 0.0L;
        
        // Reset synaptic currents
        for (size_t j = 0; j < neuron->num_inputs; j++) {
            neuron->synaptic_currents[j] = 0.0L;
        }
    }
    
    // Clear spike trains
    for (size_t i = 0; i < snn->num_neurons; i++) {
        for (size_t j = 0; j < snn->max_spike_trains_length; j++) {
            snn->spike_trains[i][j] = -1.0L;
        }
    }
}

// Get firing rate for a neuron
long double SNN_get_firing_rate(SNN_t *snn, size_t neuron_id, long double time_window) {
    if (!snn || neuron_id >= snn->num_neurons) return 0.0L;
    
    SNNNeuron *neuron = &snn->neurons[neuron_id];
    long double start_time = snn->current_time - time_window;
    long double end_time = snn->current_time;
    
    long double spike_count = 0.0L;
    for (size_t i = 0; i < snn->max_spike_trains_length; i++) {
        long double spike_time = snn->spike_trains[neuron_id][i];
        if (spike_time >= start_time && spike_time <= end_time) {
            spike_count += 1.0L;
        }
    }
    
    return spike_count / (time_window / snn->dt);
}

// Get spike times for a neuron
long double *SNN_get_spike_times(SNN_t *snn, size_t neuron_id) {
    if (!snn || neuron_id >= snn->num_neurons) return NULL;
    
    // Count actual spikes
    size_t spike_count = 0;
    for (size_t i = 0; i < snn->max_spike_trains_length; i++) {
        if (snn->spike_trains[neuron_id][i] >= 0.0L) {
            spike_count++;
        }
    }
    
    if (spike_count == 0) return NULL;
    
    long double *spike_times = malloc(spike_count * sizeof(long double));
    if (!spike_times) return NULL;
    
    size_t index = 0;
    for (size_t i = 0; i < snn->max_spike_trains_length; i++) {
        if (snn->spike_trains[neuron_id][i] >= 0.0L) {
            spike_times[index++] = snn->spike_trains[neuron_id][i];
        }
    }
    
    return spike_times;
}

// Set current-based mode
void SNN_set_current_based(SNN_t *snn, bool use_current_based) {
    if (snn) {
        snn->use_current_based = use_current_based;
    }
}

// Count total spikes in time window
size_t SNN_count_spikes(SNN_t *snn, long double time_window) {
    if (!snn) return 0;
    
    size_t total_spikes = 0;
    for (size_t i = 0; i < snn->num_neurons; i++) {
        total_spikes += SNN_count_spikes(snn, i, time_window);
    }
    return total_spikes;
}

// STDP learning
void SNN_stdp_update(SNN_t *snn, size_t pre_neuron, size_t post_neuron, 
                    long double pre_spike_time, long double post_spike_time,
                    long double A_plus, long double A_minus, long double tau_plus, long double tau_minus) {
    if (!snn || pre_neuron >= snn->num_neurons || post_neuron >= snn->num_neurons) return;
    
    if (pre_neuron >= snn->input_size || post_neuron >= snn->num_neurons) return;
    
    SNNNeuron *pre = &snn->neurons[pre_neuron];
    SNNNeuron *post = &snn->neurons[post_neuron];
    
    long double dt = snn->dt;
    long double delta_t = post_spike_time - pre_spike_time;
    
    // Calculate STDP weight update
    long double delta_w = A_plus * exp(-delta_t / tau_plus) * (1.0L - exp(-delta_t / tau_minus));
    
    if (pre_neuron < snn->input_size && post_neuron < snn->num_neurons) {
        pre->input_weights[post_neuron] += delta_w;
    }
}

// Set STDP parameters
void SNN_set_stdp_params(SNN_t *snn, long double A_plus, long double A_minus, 
                        long double tau_plus, long double tau_minus) {
    // This would typically be stored in the SNN structure
    // For now, we'll pass these parameters directly to STDP_update
    (void)A_plus, (void)A_minus, (void)tau_plus, (void)tau_minus;
}

// Print summary
void SNN_print_summary(SNN_t *snn) {
    if (!snn) return;
    
    printf("=== SNN Summary ===\n");
    printf("Neurons: %zu\n", snn->num_neurons);
    printf("Input Size: %zu\n", snn->input_size);
    printf("Output Size: %zu\n", snn->output_size);
    printf("Time Step: %.6Lf ms\n", snn->dt * 1000.0L);
    printf("Simulation Time: %.2Lf s\n", snn->simulation_time);
    printf("Current-Based Mode: %s\n", snn->use_current_based ? "Yes" : "No");
    printf("==================\n");
}

// Get spike trains for a neuron (internal function)
static long double **get_spike_trains(SNN_t *snn, size_t neuron_id) {
    if (!snn || neuron_id >= snn->num_neurons) return NULL;
    return &snn->spike_trains[neuron_id];
}
