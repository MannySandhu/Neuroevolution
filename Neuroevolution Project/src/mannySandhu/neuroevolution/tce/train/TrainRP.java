package mannySandhu.neuroevolution.tce.train;

import mannySandhu.neuroevolution.tce.TrainingStrategy;

public class TrainRP {
	/**
	 * Run experiment using the RP strategy
	 * @param args
	 */
	public static void main(String [] args){
		
		TrainingStrategy train = new TrainingStrategy();
		
		// Train the mlg network using the mlg training set
		train.RP(0.001);
		
		
		/* 
		 * Test the accuracy of the network in estimating
		 * target costs for new mlgs using the validation set
		 */
		train.testNetwork(train.data.trainingData, "Training set");
		train.testNetwork(train.data.testData, "Validation set");
	}

}
