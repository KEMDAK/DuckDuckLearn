package data;
import java.util.Arrays;
import java.util.Comparator;

public class Observation {

	public double[] data;
	public int label;
	
	public Observation(double[] data, int label) {
		this.data = new double[data.length];
		this.label = label;
		
		System.arraycopy(data, 0, this.data, 0, data.length);
	}
	
	public static Comparator<Observation> compareOn(int featureNumber) {
		return new Comparator<Observation>() {
			
			@Override
			public int compare(Observation o1, Observation o2) {
				return Double.compare(o1.data[featureNumber], o2.data[featureNumber]);
			}
		};
	}

	@Override
	public String toString() {
		return "Observation [data=" + Arrays.toString(data) + ", label=" + label + "]";
	}
}
