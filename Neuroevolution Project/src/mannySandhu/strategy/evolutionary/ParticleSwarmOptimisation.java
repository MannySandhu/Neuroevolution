package mannySandhu.strategy.evolutionary;

import org.encog.Encog;
import org.encog.mathutil.randomize.NguyenWidrowRandomizer;
import org.encog.ml.CalculateScore;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.pso.NeuralPSO;

/**
 * Class implements a Particle Swarm Algorithm
 * training strategy on a Neural Network.
 * 
 * @author Manny S
 *
 */
public class ParticleSwarmOptimisation {

	private MLTrain train = null;
	
	// Constructor
	public ParticleSwarmOptimisation(BasicNetwork network, MLDataSet trainingSet, int popSize){
		
		CalculateScore score = new TrainingSetScore(trainingSet);
		train = new NeuralPSO(
				network, new NguyenWidrowRandomizer(), score, popSize);
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
