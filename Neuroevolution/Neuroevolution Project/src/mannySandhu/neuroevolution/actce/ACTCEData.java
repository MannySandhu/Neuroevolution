package mannySandhu.neuroevolution.actce;

import org.encog.ml.data.specific.CSVNeuralDataSet;

public class ACTCEData {
	
	/**
	 * MLG training data
	 */
	public CSVNeuralDataSet trainingData = new CSVNeuralDataSet(
			"C://Users//Manny S//FYP Tools//Neuroevolution Project//res//data//mlg//mlg training set.csv",
			3, 1, true);
	
	/**
	 * MLG test data
	 */
	public CSVNeuralDataSet testData = new CSVNeuralDataSet(
			"C://Users//Manny S//FYP Tools//Neuroevolution Project//res//data//mlg//mlg test set.csv",
			3, 1, true);
	
}
