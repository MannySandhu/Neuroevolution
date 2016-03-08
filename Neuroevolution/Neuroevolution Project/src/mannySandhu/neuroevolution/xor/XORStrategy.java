package mannySandhu.neuroevolution.xor;

import org.encog.Encog;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.specific.CSVNeuralDataSet;

import mannySandhu.artificialNeuralNetwork.XORNeuralNetwork;
import mannySandhu.strategy.EvolutionaryStrategy;
import mannySandhu.strategy.evolutionary.GeneticAlgorithm;
import mannySandhu.strategy.evolutionary.ParticleSwarmOptimisation;
import mannySandhu.strategy.propagation.BackPropagation;

/**
 * Class initialises a neural network with
 * a training strategy.
 * 
 * @author Manny S
 *
 */
public class XORStrategy implements EvolutionaryStrategy {

	/**
	 * An XOR neural network
	 */
	private static XORNeuralNetwork network = new XORNeuralNetwork();

	/**
	 * Run the BP strategy
	 * @param error specifies acceptable error
	 */
	public void BP(double error){
		BackPropagation bp = new BackPropagation(
				network.network, network.trainingSet);
		bp.trainNetwork(error);
	}
	
	/**
	 * Run the GA strategy
	 * @param error specifies acceptable error
	 * @param popSize specifies the population size
	 */
	public void GA(double error, int popSize){
		GeneticAlgorithm ga = new GeneticAlgorithm(
				network.network, network.trainingSet, popSize);
		ga.trainNetwork(error);
	}
	
	/**
	 * Run the PSO strategy
	 * @param error specifies the acceptable error
	 * @param swarmSize specifies the swarm size
	 */
	public void PSO(double error, int swarmSize){
		ParticleSwarmOptimisation pso = new ParticleSwarmOptimisation(
				network.network, network.trainingSet, swarmSize);
		pso.trainNetwork(error);
	}

	public void SA(double minError) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * Compute the neural network output using a test set,
	 * training set used as test set for XOR network
	 */
	@Override
	public void testNetwork(CSVNeuralDataSet data, String TAG) {

		// Display network training results
		System.out.println("Neural Network Results: " + TAG);
		for(MLDataPair pair : network.trainingSet) {
			final MLData output = 
					network.network.compute(pair.getInput());
					
			System.out.println(pair.getInput().getData(0)
					+ "," + pair.getInput().getData(1)
					+ ", actual=" + output.getData(0) + " ,ideal=" +
					pair.getIdeal().getData(0));
		}
		Encog.getInstance().shutdown();
	}

	
}
