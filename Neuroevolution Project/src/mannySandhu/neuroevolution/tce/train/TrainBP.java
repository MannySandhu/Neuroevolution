package mannySandhu.neuroevolution.tce.train;

import mannySandhu.neuroevolution.tce.TrainingStrategy;

/**
 * Aircraft component cost estimation:
 * 
 * Using a bp training strategy to forcast target
 * costs for Main Landing Gears (MLGs).
 * 
 * @author Manny S
 *
 */
public class TrainBP {

	/**
	 * Run experiment using the BP strategy
	 * @param args
	 */
	public static void main(String [] args){
		
		TrainingStrategy train = new TrainingStrategy();
		
		// Train the mlg network using the mlg training set
		train.BP(0.001, 0.7, 0.9);
		
		
		/* 
		 * Test the accuracy of the network in estimating
		 * target costs for new mlgs using the validation set
		 */
		train.testNetwork(train.data.trainingData, "Training set");
		train.testNetwork(train.data.testData, "Validation set");
	}
	
}


