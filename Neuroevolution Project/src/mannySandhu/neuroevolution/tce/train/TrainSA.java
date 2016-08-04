package mannySandhu.neuroevolution.tce.train;

import mannySandhu.neuroevolution.tce.TrainingStrategy;

public class TrainSA {
	/**
	 * Run experiment using the SA strategy
	 * @param args
	 */
	public static void main(String [] args){
		
		TrainingStrategy train = new TrainingStrategy();
		
		// Train the mlg network using the mlg training set
		train.SA(0.001, 10, 2, 100);
	
		
		/* 
		 * Test the accuracy of the network in estimating
		 * target costs for new mlgs using the validation set
		 */
		train.testNetwork(train.data.trainingData, "Training set");
		train.testNetwork(train.data.testData, "Validation set");

	}

}
