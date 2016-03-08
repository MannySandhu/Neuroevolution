package mannySandhu.neuroevolution.actce;

import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.specific.CSVNeuralDataSet;

import mannySandhu.artificialNeuralNetwork.ACTCENeuralNetwork;
import mannySandhu.strategy.EvolutionaryStrategy;
import mannySandhu.strategy.evolutionary.GeneticAlgorithm;
import mannySandhu.strategy.evolutionary.ParticleSwarmOptimisation;
import mannySandhu.strategy.propagation.BackPropagation;

/**
 * Class initialises a neural network for training.
 * 
 * @author Manny S
 *
 */
public class ACTCEStrategy implements EvolutionaryStrategy {

	/**
	 * The MLG neural network
	 */
	public ACTCENeuralNetwork network = new ACTCENeuralNetwork();
	
	// Data object
	public ACTCEData data = new ACTCEData();
	
	@Override
	public void BP(double minError) {
		BackPropagation bp = new BackPropagation(
				network.network, data.trainingData);
		bp.trainNetwork(minError);
		
	}

	@Override
	public void GA(double minError, int popSize) {
		GeneticAlgorithm ga = new GeneticAlgorithm(
				network.network, data.trainingData, popSize);
		ga.trainNetwork(minError);
		
	}

	@Override
	public void PSO(double minError, int swarmSize) {
		ParticleSwarmOptimisation pso = new ParticleSwarmOptimisation(
				network.network, data.trainingData, swarmSize);
		pso.trainNetwork(minError);
		
	}

	@Override
	public void SA(double minError) {
		// TODO Auto-generated method stub
		
	}

	/**
	 *  Test the trained neural network on test data
	 */
	@Override
	public void testNetwork(CSVNeuralDataSet data, String TAG) {
		
		System.out.println("\nNeural Network Results: " + TAG);
		for(MLDataPair pair : data){
			final MLData output =
					network.network.compute(pair.getInput());
			
			System.out.println(pair.getInput().getData(0)
					+ "," + pair.getInput().getData(1)
					+ "," + pair.getInput().getData(2)
					
					+ ", actual=" + output.getData(0) + ", ideal=" +
					pair.getIdeal().getData(0) + ", Error=" + 
					calculateErrorPercentage(
							output.getData(0),pair.getIdeal().getData(0)) + "%");
			
		}
		//Encog.getInstance().shutdown();
	}
	
	/**
	 * Calculate the error percentage between
	 * the estimate and ideal value
	 * 
	 * @param estimate is the estimated value
	 * @param ideal is the ideal value
	 * @return the error percentage
	 */
	private double calculateErrorPercentage(
			double estimate, double ideal){
		
		// Returns the absolute value
		return Math.abs(((estimate - ideal)/ideal) * 100);
	}

}
