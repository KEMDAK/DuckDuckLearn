import java.io.IOException;

import data.CSVReader;
import data.Dataset;
import data.Observation;
import models.DecisionTree;
import models.KnnClassifier;

public class Main {

	public static void main(String[] args) throws IOException {
		CSVReader reader = new CSVReader("dataset.csv");
		
		Observation xa = new Observation(new double[] {4.1, -0.1, 2.2}, -1);
		Observation xb = new Observation(new double[] {6.1, 0.4, 1.3}, -1);
		
		int numberOfFeatures = 3;
		int numberOfClasses = 3;
		
		Dataset trainingDataset = reader.load(numberOfFeatures, numberOfClasses, true);
		
//		Dataset testingDataset = trainingDataset.trainTestSplit(0.2);
		
		DecisionTree dt = new DecisionTree(2);
		dt.fit(trainingDataset);
		KnnClassifier knn = new KnnClassifier(trainingDataset);
		
		System.out.println("DT");
		System.out.println(xa);
		System.out.println(dt.predict(xa));
		System.out.println(xb);
		System.out.println(dt.predict(xb));
		System.out.println();
		System.out.println("KNN Classification");
		System.out.println(xa);
		System.out.println(knn.mode(xa, 3));
		System.out.println(xb);
		System.out.println(knn.mode(xb, 3));
		System.out.println();
		System.out.println("KNN Regression");
		System.out.println(xa);
		System.out.println(knn.mean(xa, 3));
		System.out.println(xb);
		System.out.println(knn.mean(xb, 3));
		System.out.println();
		
//		int correctCount = 0;
//		
//		for (Observation o : testingDataset.observations) {
//			int predection = knnClassifier.mode(o, 3);
//			
//			if (o.label == predection) {
//				correctCount++;
//			}
//		}
//		
//		System.out.printf("Accuracy: %.5f", (double) correctCount / testingDataset.size);
	}
}
