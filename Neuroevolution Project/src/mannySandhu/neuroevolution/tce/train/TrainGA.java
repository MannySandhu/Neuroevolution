package mannySandhu.neuroevolution.tce.train;

import mannySandhu.neuroevolution.tce.TrainingStrategy;

public class TrainGA {
	
	public static void main(String [] args){
		
		TrainingStrategy train = new TrainingStrategy();
		
		train.GA(0.001, 100);
		
		train.testNetwork(train.data.trainingData, "Training data");
		train.testNetwork(train.data.testData, "Test data");
	}

}
