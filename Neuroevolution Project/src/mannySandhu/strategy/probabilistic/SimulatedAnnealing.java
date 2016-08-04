package mannySandhu.strategy.probabilistic;

import org.encog.Encog;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;

public class SimulatedAnnealing {
	
	// Training strategy handle, network object and training set object
		private MLTrain train = null;
		
		// Constructor
		public SimulatedAnnealing(BasicNetwork network, MLDataSet trainingSet, double startTemp, double stopTemp, int cycles){
			
			train = new NeuralSimulatedAnnealing(network, new TrainingSetScore(trainingSet), 
					startTemp, stopTemp, cycles);
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
