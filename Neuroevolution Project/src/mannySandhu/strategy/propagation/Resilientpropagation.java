package mannySandhu.strategy.propagation;

import org.encog.Encog;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class Resilientpropagation {
	
	// Training strategy handle, network object and training set object
		private ResilientPropagation train = null;
		
		// Constructor
		public Resilientpropagation(BasicNetwork network, MLDataSet trainingSet){
			
			train = new ResilientPropagation(network, trainingSet);
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
