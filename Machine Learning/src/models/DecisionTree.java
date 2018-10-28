package models;
import java.util.Arrays;

import data.Dataset;
import data.Observation;

public class DecisionTree { 
	
	private static class Node {
		Node left;
		Node right;
	}
	
	private static class TreeNode extends Node{
		int featureNumber;
		double threshold;
		
		public TreeNode(int featureNumber, double threshold) {
			this.featureNumber = featureNumber;
			this.threshold = threshold;
		}

		@Override
		public String toString() {
			return "TreeNode [featureNumber = X" + featureNumber + ", threshold=" + threshold + "]";
		}
	}
	
	private static class LeafNode extends Node{
		Integer[] classFrequency;
		
		public LeafNode(Integer[] frequencies) {
			this.classFrequency = frequencies;
		}
		
		public boolean isPure() {
			int positiveCount = 0;
			for (Integer x : classFrequency) {
				positiveCount += x > 0? 1 : 0;
			}
			
			return positiveCount == 1;
		}

		@Override
		public String toString() {
			return "LeafNode [classFrequency=" + Arrays.toString(classFrequency) + "]";
		}
	}
	
	private int depth;
	private Node root;
	
	public DecisionTree(int depth) {
		this.depth = depth;
	}
	
	public void fit(Dataset trainingData) {
		this.root = new LeafNode(trainingData.getFrequencies());
		
		this.root = fit((LeafNode) this.root, trainingData, 0);
	}
	
	private Node fit(LeafNode currentNode, Dataset dataset, int depth) {
		double currentImpurityMeasure = giniIndex(currentNode.classFrequency); // this can be any Impurity Measure  

		if (depth == this.depth) {
			return currentNode;
		}
		
		if (currentNode.isPure()) {
			return currentNode;
		}
		
		Dataset BestLeftDataset = new Dataset(dataset.numberOfFeatures, dataset.numberOfClasses);
		Dataset BestRightDataset = new Dataset(dataset.numberOfFeatures, dataset.numberOfClasses);
		double bestImprovement = 0;
		
		int bestFeatureNumber = 0; // The feature that we will split on
		double bestThreshold = 0; // The threshold that we will split on
		
		for (int i = 0; i < dataset.numberOfFeatures; i++) {
			// sorting the data on the ith dimension
			dataset.sort(i);
		
			Dataset leftDataset = new Dataset(dataset.numberOfFeatures, dataset.numberOfClasses);
			Dataset rightDataset = dataset.clone();
			
			while (!rightDataset.isEmpty()) {
				leftDataset.addLast(rightDataset.removeFirst());
				
				if (!rightDataset.isEmpty() && leftDataset.peekLast() == rightDataset.peekFirst()) {
					continue;
				}
				
				double leftPortion = (double) leftDataset.size / dataset.size;
				double leftImpurityMeasure = giniIndex(leftDataset.getFrequencies());
				
				double rightPortion = (double) rightDataset.size / dataset.size;
				double rightImpurityMeasure = giniIndex(rightDataset.getFrequencies());
				
				double improvement = currentImpurityMeasure - (leftPortion * leftImpurityMeasure) - (rightPortion * rightImpurityMeasure);
				
				if (improvement > bestImprovement) {
					BestLeftDataset = leftDataset.clone();
					BestRightDataset = rightDataset.clone();
					bestImprovement = improvement;
					bestFeatureNumber = i;
					bestThreshold = leftDataset.peekLast().data[i];
				}
			}
		}
		
		TreeNode currentTreeNode = new TreeNode(bestFeatureNumber, bestThreshold);
		
		currentTreeNode.left = fit(new LeafNode(BestLeftDataset.getFrequencies()), BestLeftDataset, depth + 1);
		currentTreeNode.right = fit(new LeafNode(BestRightDataset.getFrequencies()), BestRightDataset, depth + 1);
		
		return currentTreeNode;
	}

	private double giniIndex(Integer[] frequencies) {
		final int total = Arrays.stream(frequencies).reduce(0, (x, y) -> x + y);
		final int totalSq = total * total;
		return 1 - Arrays.stream(frequencies)
				.map(x -> x * x)
				.mapToDouble(x -> x * 1.0)
				.map(x -> x / totalSq)
				.reduce(0, (x, y) -> x + y);
	}
	
	public int predict(Observation observation) {
		return predictHelper(observation, this.root);
	}

	private int predictHelper(Observation observation, Node currentNode) {
		if (currentNode instanceof LeafNode) {
			int maxIdx = 0;
			Integer[] frequencies = ((LeafNode) currentNode).classFrequency;
			
			for (int i = 1; i < frequencies.length; ++i) {
				if (frequencies[i] > frequencies[maxIdx]) {
					maxIdx = i;
				}
			}
			
			return maxIdx;
		}
		
		TreeNode treeNode = (TreeNode)currentNode;
		if (observation.data[treeNode.featureNumber] <= treeNode.threshold) {
			return predictHelper(observation, treeNode.left);
		} else {
			return predictHelper(observation, treeNode.right);
		}
	}
	
	@Override
	public String toString() {
		return trace(this.root, 0);
	}

	private String trace(Node currentNode, int level) {
		if (currentNode instanceof LeafNode) {
			return currentNode.toString();
		}
		String res = "T" + level + "( " + currentNode + ", ";
		res += trace(currentNode.left, level + 1) + ", ";
		res += trace(currentNode.right, level + 1);
		return res + ") ";
	}
}
