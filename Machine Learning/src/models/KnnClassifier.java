package models;
import java.util.Collections;

import data.Dataset;
import data.Observation;

public class KnnClassifier {

	Dataset dataset;
	
	public KnnClassifier(Dataset dataset) {
		this.dataset = dataset;
	}
	
	public int mode(Observation observation, int k) {
		Observation[] nearestNeighbors = getNearestNeighbors(observation, k);
		int freq[] = new int[dataset.numberOfClasses];
		
		int maxIdx = 0;
		
		for (int i = 0; i < k; ++i) {
			
			freq[nearestNeighbors[i].label]++;
			
			if (freq[nearestNeighbors[i].label] > freq[maxIdx]) {
				maxIdx = nearestNeighbors[i].label;
			}
		}
		
		return maxIdx;
	}
	
	public double mean(Observation observation, int k) {
		Observation[] nearestNeighbors = getNearestNeighbors(observation, k);
		
		double label = 0;
		
		for (int i = 0; i < k; ++i) {
			label += (double) nearestNeighbors[i].label / k;
		}
		
		return label;
	}
	
	public Observation[] getNearestNeighbors(Observation observation, int k) {
		Collections.sort(dataset.observations, (ob1, ob2) -> {
			return Long.compare(squaredEuclideanDistance(ob1, observation),
					squaredEuclideanDistance(ob2, observation));
		});
		
		Observation[] nearestNeighbors = new Observation[k];
		
		int i = 0;
		for (Observation next : dataset.observations) {
			if (i == k) {
				break;
			}
			nearestNeighbors[i] = next;
			i++; 
		}
		
		return nearestNeighbors;
	}
	
	private long squaredEuclideanDistance(Observation observation1, Observation observation2) {
		long res = 0;
		
		for (int i = 0; i < dataset.numberOfFeatures; ++i) {
			res += 1L * (observation1.data[i] - observation2.data[i]) * 
					(observation1.data[i] - observation2.data[i]);
		}
		
		return res;
	}
}
