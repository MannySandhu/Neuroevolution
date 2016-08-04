package mannySandhu.neuroevolution.xor;

/**
 * Class for training and testing the
 * XOR neural network once initialised.
 * 
 * @author Manny S
 *
 */
public class ANNTrain {
	
	/**
	 * Trainer object initialised with the network and
	 * training set
	 */
	private static TrainingStrategy xorStrategy = new TrainingStrategy();
	
	/**
	 * The main method
	 * @param args takes no input
	 */
	public static void main(String [] args){
		
		xorStrategy.BP(0.01, 0.25, 0);
		//strategy.runGA(0.01, 100);
		//strategy.runPSO(0.01, 20);
		xorStrategy.testNetwork(null, "XOR");
	}

}
