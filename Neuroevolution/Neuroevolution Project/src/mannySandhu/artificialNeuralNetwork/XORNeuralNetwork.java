package mannySandhu.artificialNeuralNetwork;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;

/**
 * A simple class to construct a XOR Neural Network
 * with input and ideal output used as the training
 * set. Training strategy is defined in the specific
 * implementation of this network.
 * 
 * @author Manny S
 *
 */
public class XORNeuralNetwork {
	
	/**
	 * Define the input and ideal output
	 */
	// XOR function input
	private double XOR_INPUT[][] = {
			{0.0, 0.0},
			{1.0, 0.0},
			{0.0, 1.0},
			{1.0, 1.0}
	};
	// XOR function ideal output
	private double XOR_IDEAL[][] = {
			{0.0},
			{1.0},
			{1.0},
			{0.0}
	};
	/**
	 * Define a basic network and training set
	 */
	// Neural network
	public BasicNetwork network = new BasicNetwork();

	// Training set
	public MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
	
	// Constructor
	public XORNeuralNetwork(){
		/**
		 * Create a new XOR network 
		 */
		// Define the network architecture
		// Input layer
		network.addLayer(new BasicLayer(null, true, 2));
		// Hidden layer
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
		// Output layer
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
		// Finalise network structure
		network.getStructure().finalizeStructure();
		// Set random weight values initially
		network.reset();
	}
}
