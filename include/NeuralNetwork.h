#pragma once

#include <iostream>
#include <vector>

class Neuron {
public:
	Neuron(int n_weights);
	~Neuron();

	void activate(std::vector<float> inputs);
	void transfer();
	float transfer_derivative() { return static_cast<float>(m_output * (1.0 - m_output));  };

	// return mutable reference to the neuron weights
	std::vector<float>& get_weights(void) { return m_weights; };

	float get_output(void) { return m_output; };
	float get_activation(void) { return m_activation; };
	float get_delta(void) { return m_delta; };

	void set_delta(float delta) { m_delta = delta; };

private:
	size_t m_nWeights;
	std::vector<float> m_weights;
	float m_activation;
	float m_output;
	float m_delta;

private:
	void initWeights(int n_weights);
};

class Layer {
public:
	Layer(int n_neurons, int n_weights);
	~Layer();

	// return mutable reference to the neurons
	std::vector<Neuron>& get_neurons(void) { return m_neurons; };

private:
	void initNeurons(int n_neurons, int n_weights);

	std::vector<Neuron> m_neurons;
};

class Network {
public:
	Network();
	~Network();

	void initialize_network(int n_inputs, int n_hidden, int n_outputs);

	void add_layer(int n_neurons, int n_weights);
	std::vector<float> forward_propagate(std::vector<float> inputs);
	void backward_propagate_error(std::vector<float> expected);
	void update_weights(std::vector<float> inputs, float l_rate);
	void train(std::vector<std::vector<float>>trainings_data, float l_rate, size_t n_epoch, size_t n_outputs);
	int predict(std::vector<float> input);

	void display_human();

private:
	size_t m_nLayers;
	std::vector<Layer> m_layers;

};
