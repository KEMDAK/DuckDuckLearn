package data;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CSVReader {

	private String filePath;
	
	public CSVReader(String filePath) {
		this.filePath = filePath;
	}
	
	public Dataset load(int numberOfFeatures, int numberOfClasses, boolean header) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		
		if (header) {
			reader.readLine();
		}
		
		Dataset dataset = new Dataset(numberOfFeatures, numberOfClasses);
		List<Observation> observations = new ArrayList<>();
		
		while(reader.ready()) {
			String[] line = reader.readLine().replaceAll(" ", "").split(",");
			
			try {
				double[] data = new double[numberOfFeatures];
				
				for (int i = 0; i < numberOfFeatures; ++i) {
					data[i] = Double.parseDouble(line[i]);
				}
				
				observations.add(new Observation(data, Integer.parseInt(line[numberOfFeatures])));
			} catch (Exception e) {
				System.out.println("Error Parsing data: " + Arrays.toString(line));
			}
		}
		
		dataset.addCollection(observations);
		
		reader.close();
		
		return dataset;
	}
}
