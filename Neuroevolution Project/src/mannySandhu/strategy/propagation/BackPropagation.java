package mannySandhu.strategy.propagation;

import org.encog.Encog;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.propagation.back.Backpropagation;

/**
 * A class to implement a back propagation training 
 * strategy for neural networks.
 * 
 * @author Manny S
 *
 */
public class BackPropagation {

	// Training strategy handle, network object and training set object
	private MLTrain train = null;
	
	// Constructor
	public BackPropagation(BasicNetwork network, MLDataSet trainingSet, double learningRate, double momentum){
		
		train = new Backpropagation(network, trainingSet, learningRate, momentum);
	}
	
	/**
	 * Train the neural network using the strategy
	 * @param acceptableError specifies acceptable error 
	 */
	public void trainNetwork(double acceptableError){
		
		// Train the network until error <= acceptable error
		int epoch = 1;
		do {
			train.iteration();
			System.out.println("Epoch #" + epoch + " Error:" + train.getError());
			++epoch;
		}
		while(train.getError() > acceptableError);
		train.finishTraining();
	
		Encog.getInstance().shutdown();
	}
}
