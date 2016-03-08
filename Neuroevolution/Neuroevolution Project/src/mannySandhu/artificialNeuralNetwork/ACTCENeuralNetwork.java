package mannySandhu.artificialNeuralNetwork;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;

/**
 * 
 * Artificial Neural Network for Regression Analysis
 * 
 * A Neural Network for target cost estimation of MLGs 
 * (main landing gear). Given the three parameters, 
 * the MLG cost drivers, the network estimates target 
 * costs of an MLG with unknown input values.
 * 
 * @author Manny S
 *
 */
public class ACTCENeuralNetwork {
	
	/*
	 * The Neural network object
	 */
	public BasicNetwork network = new BasicNetwork();
	
	/**
	 * Network Architecture:
	 * 
	 * A fully connected, feed-forward, Multi-layer Perceptron
	 * consisting of three layers, three neurons in
	 * the input layer with a bias neuron, five in 
	 * the hidden layer with a bias neuron and one
	 * output neuron. A non-linear activation function,
	 * the Sigmoid function.  
	 */
	public ACTCENeuralNetwork(){
		
		/*
		 * Input layer with three input neurons and a bias
		 * neuron with no activation function.
		 */
		network.addLayer(new BasicLayer(null, true, 3));
		
		/*
		 * Hidden layer with five hidden neurons and a bias neuron
		 * with Sigmoid activation function.
		 */
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 5));
		
		/*
		 * Output layer with one neuron and Sigmoid activation
		 * function.
		 */
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
		
		/*
		 * Finalise the network structure
		 */
		network.getStructure().finalizeStructure();
		
		/*
		 * Set initial weights to random
		 */
		network.reset();
	}
}
