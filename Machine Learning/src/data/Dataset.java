package data;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedList;

public class Dataset {
	
	public int size;
	public int numberOfFeatures;
	public int numberOfClasses;
	public LinkedList<Observation> observations;
	
	public Dataset(int numberOfFeatures, int numberOfClasses) {
		this.size = 0;
		this.numberOfFeatures = numberOfFeatures;
		this.numberOfClasses = numberOfClasses;
		this.observations = new LinkedList<Observation>();
	}
	
	public Dataset clone() {
		Dataset result = new Dataset(numberOfFeatures, numberOfClasses);
		result.addCollection(observations);
		
		return result;
	}
	
	public void sort(int featureNumber) {
		Collections.sort(observations, Observation.compareOn(featureNumber));
	}
	
	public void addCollection(Collection<Observation> observations) {
		this.observations.addAll(observations);
		this.size += observations.size();
	}
	
	public void addFirst(Observation observation) {
		this.observations.addFirst(observation);
		this.size++;
	}
	
	public void addLast(Observation observation) {
		this.observations.addLast(observation);
		this.size++;
	}
	
	public Observation removeFirst() {
		this.size--;
		return this.observations.removeFirst();
	}
	
	public Observation removeLast() {
		this.size--;
		return this.observations.removeLast();
	}
	
	public Observation peekFirst() {
		return this.observations.peekFirst();
	}
	
	public Observation peekLast() {
		return this.observations.peekLast();
	}
	
	public boolean isEmpty() {
		return size == 0;
	}
	
	public Integer[] getFrequencies() {
		Integer[] result = new Integer[numberOfClasses];
		Arrays.fill(result, 0);
		
		for (Observation observation : observations) {
			result[observation.label]++;
		}
		
		return result;
	}
	
	private void shuffle() {
		Collections.shuffle(observations);
	}
	
	public Dataset trainTestSplit(double testPercentage) {
		shuffle();

		Dataset testingDataset = new Dataset(numberOfFeatures, numberOfClasses);
		
		int testingSize = (int) (testPercentage * size);
		for (int i = 0; i < testingSize; i++) {
			testingDataset.addLast(this.removeLast());
		}
		
		return testingDataset;
	}
}
