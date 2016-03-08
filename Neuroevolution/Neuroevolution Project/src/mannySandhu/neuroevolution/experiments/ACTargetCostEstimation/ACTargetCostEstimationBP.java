package mannySandhu.neuroevolution.experiments.ACTargetCostEstimation;

import mannySandhu.neuroevolution.actce.ACTCEStrategy;

/**
 * Aircraft component cost estimation:
 * 
 * Using a bp training strategy to forcast target
 * costs for Main Landing Gears (MLGs).
 * 
 * @author Manny S
 *
 */
public class ACTargetCostEstimationBP {

	/**
	 * Run experiment using the BP strategy
	 * @param args
	 */
	public static void main(String [] args){
		
		ACTCEStrategy mlg = new ACTCEStrategy();
		
		// Train the mlg network using the mlg training set
		mlg.BP(0.01);
		
		/*
		 * Test the network on the training set first
		 */
		mlg.testNetwork(mlg.data.trainingData, "Training set");
		
		/* 
		 * Test the accuracy of the network in estimating
		 * target costs for new mlgs using the validation set
		 */
		mlg.testNetwork(mlg.data.testData, "Validation set");
	}
	
}
