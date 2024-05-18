#include "NeuralNetwork.h"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>

/* NEURON */

/*
* Neuron Constructor
*/
Neuron::Neuron(int n_weights) {
	this->initWeights(n_weights);
	m_nWeights = n_weights;
	m_activation = 0;
	m_output = 0;
	m_delta = 0;
}

/*
* Neuron Destructor
*/
Neuron::~Neuron() {
	/*pass*/
}

/*
* Initialize weights
*/
void Neuron::initWeights(int n_weights) {
	// add random small weights (between 0 and 1)
	for (int w = 0; w < n_weights; w++) {
		m_weights.push_back(static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
	}
}

/* 
* Calculate the activation of a neuron for a given input
*/
void Neuron::activate(std::vector<float> inputs) {
	// the last weight is assumed to be the bias
	m_activation = m_weights[m_nWeights-1];

	// accumulate all weighted inputs
	for (size_t i = 0; i < m_nWeights-1; i++)
	{
		m_activation += m_weights[i] * inputs[i];
	}
}

/* 
* Transfer the activation of the neuron to an actual output
*/
void Neuron::transfer() {
	m_output = 1.0f / (1.0f + std::exp(-m_activation));
}

/* LAYER */

/*
* Layer Constructor
*/
Layer::Layer(int n_neurons, int n_weights) {
	this->initNeurons(n_neurons, n_weights);
}

/*
* Layer Destructor
*/
Layer::~Layer() {
	/*pass*/
}

void Layer::initNeurons(int n_neurons, int n_weights) {
	for (int n = 0; n < n_neurons; n++) {
		m_neurons.push_back(Neuron(n_weights));
	}
}


/* NETWORK */

/*
* Network Constructor
*/
Network::Network() {
	// initialize prng
	std::srand(static_cast<unsigned int>(std::time(nullptr)));

	m_nLayers = 0;
}

/* 
* Network Destructor
*/
Network::~Network() {
	/*pass*/
}

/* 
* Initialize a network manually
*/
void Network::initialize_network(int n_inputs, int n_hidden, int n_outputs) {

	// add a hidden layer (n_hidden neurons are each connected to all inputs)
	this->add_layer(n_hidden, n_inputs+1);

	// add an output layer (one neuron for each output is connected to all neurons from the previous layer)
	this->add_layer(n_outputs, n_hidden+1);
}

/*
* Add another layer to the network
*/
void Network::add_layer(int n_neurons, int n_weights) {
	m_layers.push_back(Layer(n_neurons, n_weights));
	m_nLayers++;
}

/*  
* One forward propagation of an input
*/
std::vector<float> Network::forward_propagate(std::vector<float> inputs) {
	std::vector<float> new_inputs;
	for (size_t i = 0; i < m_nLayers; i++)
	{
		new_inputs.clear();
		
		// reference the layer neurons directly
		std::vector<Neuron>& layer_neurons = m_layers[i].get_neurons();
		for (size_t n = 0; n < layer_neurons.size(); n++)
		{
			layer_neurons[n].activate(inputs);
			layer_neurons[n].transfer();
			new_inputs.push_back(layer_neurons[n].get_output());
		}
		inputs = new_inputs;
	}
	return inputs;
}

/* 
* Propagate the deviation from an expected output backwards through the network
*/
void Network::backward_propagate_error(std::vector<float> expected) {
	// reverse traverse the layers
	for (size_t i = m_nLayers; i --> 0;)
	{
		// get a reference to the neurons of this layer
		std::vector<Neuron>& layer_neurons = m_layers[i].get_neurons();

		// iterate over each neuron in this layer
		for (size_t n = 0; n < layer_neurons.size(); n++)
		{
			float error = 0.0;
			// feed the expected result to the output layer
			if (i == m_nLayers - 1)
			{
				error = expected[n] - layer_neurons[n].get_output();
			}	
			else {
				for (auto& neu : m_layers[i + 1].get_neurons()) {
					error += (neu.get_weights()[n] * neu.get_delta());
				}
			}
			// update the delta value of the neuron
			layer_neurons[n].set_delta(error * layer_neurons[n].transfer_derivative());
		}
	}
}

/*
* Update weights of a network after an error back propagation
*/
void Network::update_weights(std::vector<float> inputs, float l_rate) {
	// iterate over the layers
	for (size_t i = 0; i < m_nLayers; i++)
	{		
		std::vector<float> new_inputs = {};
		if (i != 0) {
			// grab the outputs from the previous layer (except for the first layer)
			for (auto &neuron: m_layers[i-1].get_neurons())
			{
				new_inputs.push_back(neuron.get_output());
			}
		}
		else {
			// use the original input for the first layer (ignore the bias input / last element)
			new_inputs = std::vector<float>(inputs.begin(), inputs.end() - 1);
		}
		
		// get a reference to the neurons of this layer
		std::vector<Neuron>& layer_neurons = m_layers[i].get_neurons();

		for (size_t n = 0; n < layer_neurons.size(); n++)
		{
			// get a reference to the weights of the neuron
			std::vector<float>& weights = layer_neurons[n].get_weights();
			// update weights
			for (size_t j = 0; j < new_inputs.size(); j++)
			{
				weights[j] += l_rate * layer_neurons[n].get_delta() * new_inputs[j];
			}
			// update bias
			weights.back() += l_rate * layer_neurons[n].get_delta();
		}
	}
}

/*  
* Train the network with trainings data
*/
void Network::train(std::vector<std::vector<float>>trainings_data, float l_rate, size_t n_epoch, size_t n_outputs) {
	for (size_t e = 0; e < n_epoch; e++)
	{
		float sum_error = 0;

		for (const auto &row: trainings_data)
		{
			std::vector<float> outputs = this->forward_propagate(row);
			std::vector<float> expected(n_outputs, 0.0);
			expected[static_cast<int>(row.back())] = 1.0;
			for (size_t x = 0; x < n_outputs; x++)
			{
				sum_error += static_cast<float>(std::pow((expected[x] - outputs[x]), 2));
			}
			this->backward_propagate_error(expected);
			this->update_weights(row, l_rate);
		}
		std::cout << "[>] epoch=" << e << ", l_rate=" << l_rate << ", error=" << sum_error << std::endl;
	}
}

/* 
* Make a prediction for an input (one forward propagation)
*/
int Network::predict(std::vector<float> input) {
	std::vector<float> outputs = this->forward_propagate(input);
	return std::max_element(outputs.begin(), outputs.end()) - outputs.begin();
}

/*
* Display the network in a human readable format
*/
void Network::display_human() {
	std::cout << "[Network] (Layers: " << m_nLayers << ")" << std::endl;

	std::cout << "{" << std::endl;
	for (size_t l = 0; l < m_layers.size(); l++)
	{
		Layer layer = m_layers[l];
		std::cout << "\t (Layer " << l << "): {";
		for (size_t i = 0; i < layer.get_neurons().size(); i++)
		{
			Neuron neuron = layer.get_neurons()[i];
			std::cout << "<(Neuron " << i << "): [ weights={";
			std::vector<float> weights = neuron.get_weights();
			for (size_t w = 0; w < weights.size(); ++w)
			{
				std::cout << weights[w];
				if (w < weights.size() - 1) {
					std::cout << ", ";
				}
			}
			std::cout << "}, output=" << neuron.get_output() << ", activation=" << neuron.get_activation() << ", delta=" << neuron.get_delta();
			std::cout << "]>";
			if (i < layer.get_neurons().size() - 1) {
				std::cout << ", ";
			}
		}
		std::cout << "}";
		if (l < m_layers.size() - 1) {
			std::cout << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << "}" << std::endl;
}
