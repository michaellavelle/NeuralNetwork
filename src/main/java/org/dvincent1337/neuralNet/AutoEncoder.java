package org.dvincent1337.neuralNet;

import java.util.Vector;

import org.jblas.DoubleMatrix;

/**
 * Initial Auto-Encoder implementation ( without Sparsity constraints for now) and a single hidden layer
 * of configurable number of neurons
 * 
 * 
 * @author Michael Lavelle
 *
 */
public class AutoEncoder extends NeuralNetwork {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public AutoEncoder() {
		super();
	}

	/**
	 * 
	 * @param inputNeurons The number of features of each element of the input data set
	 * @param hiddenNeurons The number of compressed features that we would like the auto-encoder to use.
	 */
	public AutoEncoder(int inputNeurons,int hiddenNeurons) {
		super(new int[] {inputNeurons,hiddenNeurons,inputNeurons}, true);
	}


	
	
	/**
	 * Returns the first-level activation hypothesis of a neural network given weight matricies (Theta) and inputs (X)
	 * Uses forward propagation alrogrithm. 
	 * http://en.wikipedia.org/wiki/Feedforward_neural_network
	 */
	public static DoubleMatrix forwardPropPredictFirstActivationOnly(Vector<DoubleMatrix> Theta, DoubleMatrix X)
	{
		int m = X.getRows();
		Vector<DoubleMatrix> activations = new Vector<DoubleMatrix>(Theta.size()+1);
		
		DoubleMatrix firstActivation = new DoubleMatrix(m,Theta.firstElement().getColumns() );
		firstActivation = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(m,1), X);
		activations.add(firstActivation);
		DoubleMatrix hypothesis = new DoubleMatrix(m,Theta.firstElement().getColumns());
		hypothesis = sigmoid(firstActivation.mmul(Theta.firstElement().transpose()));
		
		return hypothesis;
	}
	

}
