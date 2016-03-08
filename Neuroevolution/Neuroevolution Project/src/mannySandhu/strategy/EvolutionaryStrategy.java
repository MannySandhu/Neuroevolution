package mannySandhu.strategy;

import org.encog.ml.data.specific.CSVNeuralDataSet;

/**
 * Interface implements evolutionary
 * strategy methods to the implementing
 * class.
 * 
 * Useful for a specific neural network
 * implementation where the strategy needs
 * to be tuned before being used in training.
 * 
 * The abstract methods must be provided an
 * implementation in the implementing class.
 * 
 * A method to implement a test is provided.
 * 
 * @author Manny S
 *
 */
public interface EvolutionaryStrategy {

	/**
	 * Provides methods for the implementation 
	 * of training strategies.
	 */
	
	/*
	 *  BP not an evolutionary strategy - 
	 *  traditional network training strategy
	 */
	void BP(double minError);
	
	void GA(double minError, int popSize);
	
	void PSO(double minError, int swarmSize);
	
	void SA(double minError);
	
	// Method implements a network test
	void testNetwork(CSVNeuralDataSet data, String TAG);
	
}
