package mannySandhu.neuroevolution.tce;

import org.encog.ml.data.specific.CSVNeuralDataSet;

public class MlgData {
	
	/**
	 * Main landing gear training and validation data
	 */
	public CSVNeuralDataSet trainingData = new CSVNeuralDataSet(
			"C://Users//Manny S//FYP Tools//Neuroevolution Project//res//data//mlg//case 1//training1.csv",
			3, 1, true);
	
	public CSVNeuralDataSet testData = new CSVNeuralDataSet(
			"C://Users//Manny S//FYP Tools//Neuroevolution Project//res//data//mlg//case 1//test1.csv",
			3, 1, true);
	
}
