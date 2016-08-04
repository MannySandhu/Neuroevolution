package mannySandhu.strategy.evolutionary;

import org.encog.Encog;
import org.encog.ml.MLMethod;
import org.encog.ml.MethodFactory;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.genetic.MLMethodGeneticAlgorithm;
import org.encog.ml.genetic.crossover.Splice;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.TrainingSetScore;

/**
 * A class that implements a genetic algorithm
 * to train a neural network.
 * 
 * @author Manny S
 *
 */
public class GeneticAlgorithm {

	// Training strategy handle, network object, training set object and population size variable
	private MLTrain trainer = null;
	
	// Constructor
	public GeneticAlgorithm(BasicNetwork network, MLDataSet trainingSet, int popSize){
		
		// Genetic algorithm initialised with the network, score function and a population size 
		trainer = new MLMethodGeneticAlgorithm(new MethodFactory(){
			
			@Override
			public MLMethod factor() {
				return network;
			}
		}, new TrainingSetScore(trainingSet), popSize);
		trainer.setIteration(1000);
	}
	

	/**
	 * Train the neural network using the strategy
	 * @param acceptableError specifies acceptable error 
	 */
	public void trainNetwork(double acceptableError){
		
		// Train the network until error <= acceptable error
		int epoch = 1;
		do {
			trainer.iteration();
			System.out.println("Epoch #" + epoch + " Error:" + trainer.getError());
			++epoch;
		}
		while(trainer.getError() > acceptableError);
		trainer.finishTraining();
		
		Encog.getInstance().shutdown();
	}
}
