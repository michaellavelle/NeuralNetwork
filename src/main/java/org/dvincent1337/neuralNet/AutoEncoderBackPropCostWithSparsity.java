package org.dvincent1337.neuralNet;

import java.util.Collections;
import java.util.Vector;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
/**
 * Customised BackPropCost which takes into account sparsity constraints.
 * 
 * This is an initial implementation which needs to be properly tested and also to be converted
 * to a vectorised implementation
 * 
 * @author Michael Lavelle
 *
 */
public class AutoEncoderBackPropCostWithSparsity extends BackPropCost {

	private DoubleMatrix X; // Training input matrix
	private DoubleMatrix Y; // Training output matrix
	private int[] topology; // Neural network topology
	private double lambda; // Used for regularization
	private double sparsityParameter;
	private double sparsityBeta;

	public AutoEncoderBackPropCostWithSparsity(DoubleMatrix setX, DoubleMatrix setY, int[] setTopology,
			double setLambda, double sparsityParameter, double sparsityBeta) {
		super(setX, setY, setTopology, setLambda);
		this.X = setX;
		this.Y = setY;
		this.topology = setTopology;
		this.lambda = setLambda;
		this.sparsityParameter = sparsityParameter;
		this.sparsityBeta = sparsityBeta;
	}

	public AutoEncoderBackPropCostWithSparsity(DoubleMatrix setX, DoubleMatrix setY, int[] setTopology, double setLambda) {
		super(setX, setY, setTopology, setLambda);
	}

	private int getHiddenNeuronsCount() {
		return topology[1];
	}

	/**
	 * -Computes the cost of given input matrix. The given input matrix should
	 * be a column matrix of all the neural network weights (possibly done with
	 * NeuralNetwork.reshapeToVector ) -Computes the partial derivatives of each
	 * element of the input matrix using backprop algorithm
	 * http://en.wikipedia.org/wiki/Backpropagation
	 * 
	 * Prototyped with matlab initially (hence the comments with matlab code.)
	 * This is a vectorized implementation (fully utilizing linear algebra
	 * instead of using unnecessary loops)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public Tuple<Double, DoubleMatrix> evaluateCost(DoubleMatrix input) {
		int num_layers = topology.length;
		Vector<DoubleMatrix> Theta = NeuralNetwork.reshapeToList(input, topology);
		int m = X.getRows();

		// ----------------|START FORWARD PROP AND FIND COST |-------------

		DoubleMatrix H = NeuralNetwork.forwardPropPredict(Theta, X);

		// matlab: J_part =( sum((Y.*((-1)*log(h)) - ((1-Y).*log(1-h))),2) );
		DoubleMatrix J_part = new DoubleMatrix(m, 1); // Cost without
														// regularization
		J_part = (Y.mul(-1).mul(MatrixFunctions.log(H))
				.sub(Y.mul(-1).add(1).mul(MatrixFunctions.log(H.mul(-1).add(1))))).rowSums();
		double ThetaReg = 0;

		// Calculate regularization part of cost.
		for (int i = 0; i < (num_layers - 1); i++) {
			DoubleMatrix currentTheta = Theta.get(i);
			int[] rows = new int[currentTheta.getRows()];
			int[] cols = new int[currentTheta.getColumns() - 1];
			for (int j = 0; j < currentTheta.getRows(); j++) {
				rows[j] = j;
			}
			for (int j = 1; j < currentTheta.getColumns(); j++) {
				cols[j - 1] = j;
			}
			ThetaReg += MatrixFunctions.pow(currentTheta.get(rows, cols), 2).sum();
		}

		double J = (J_part.sum() + (lambda) * ThetaReg) / (2 * m) + getSparsityCost(X, Theta, getHiddenNeuronsCount()); // Add
																														// the
																														// non
																														// regularization
																														// and
																														// regularization
																														// cost
																														// together

		// ----------------|FINISHED FORWARD PROP AND FOUND COST |-------------

		// ----------------|START BACK PROP AND FIND GRADIANTS |-------------

		Vector<DoubleMatrix> a = new Vector<DoubleMatrix>(); // Activations for
																// each layer
		Vector<DoubleMatrix> z = new Vector<DoubleMatrix>(); // sigmoid of z are
																// the
																// activations
																// for each
																// layer

		// Get the first Activations
		DoubleMatrix firstActivation = new DoubleMatrix(m, Theta.firstElement().getColumns());
		firstActivation = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(m, 1), X);
		a.add(firstActivation); // a{1}
		z.add(firstActivation.mmul(Theta.get(0).transpose())); // z{2}Technically
		// Get the hidden Activations
		for (int i = 1; i < (num_layers - 1); i++) {
			DoubleMatrix newa = new DoubleMatrix().copy(NeuralNetwork.sigmoid(z.get(i - 1)));
			a.add(DoubleMatrix.concatHorizontally(DoubleMatrix.ones(newa.getRows(), 1), newa));
			z.add(a.lastElement().mmul(Theta.get(i).transpose()));
		}
		a.add(NeuralNetwork.sigmoid(z.lastElement()));

		// With the DELTA and delta Lists, the first element corresponds to the
		// last layer.

		Vector<DoubleMatrix> delta = new Vector<DoubleMatrix>(); // Error for
																	// each
																	// neuron on
																	// each
																	// layer
		Vector<DoubleMatrix> DELTA = new Vector<DoubleMatrix>(); // A piece of
																	// the
																	// partial
																	// derivative
																	// of each
																	// weight
																	// matrix
		// backprop on output layer
		delta.add(new DoubleMatrix().copy((a.lastElement().sub(Y))));
		// backprop on hidden layers
		for (int i = (num_layers - 2); i >= 1; i--) {
			DoubleMatrix newDelta = new DoubleMatrix();

			DoubleMatrix m1 = (Theta.get(i).transpose().mmul(delta.firstElement().transpose())).mul(
					NeuralNetwork.sigmoidGradiant(
							DoubleMatrix.concatHorizontally(DoubleMatrix.ones(z.get(i - 1).getRows()), z.get(i - 1)))
							.transpose()).transpose();

			if (i == 1) {
				m1 = m1.add(getSparsityGradientTerm(X, Theta));
			}

			newDelta.copy(m1);// matlab: newDelta = ( (
								// ((Theta{i})')*(delta{p-1})'
								// ).*sigmoidGradient([ones(size(z{p},1),1)
								// z{p}])'

			int[] rows = new int[newDelta.getRows()];
			int[] cols = new int[newDelta.getColumns() - 1];
			for (int j = 0; j < newDelta.getRows(); j++) {
				rows[j] = j;
			}
			for (int j = 1; j < newDelta.getColumns(); j++) {
				cols[j - 1] = j;
			}

			Vector<DoubleMatrix> temp = new Vector<DoubleMatrix>((Vector<DoubleMatrix>) delta.clone());
			delta.removeAllElements();
			delta.add(newDelta.get(rows, cols));
			delta.addAll(temp);

			DELTA.add(delta.get(1).transpose().mmul(a.get(i)));
		}
		DELTA.add(delta.firstElement().transpose().mmul(a.firstElement()));

		// newDelta = newDelta.add(this.getSparsityGradientTerm(X, Theta, 4));
		// Calculate the gradients of each weight matrix
		Collections.reverse(DELTA);
		Vector<DoubleMatrix> gradList = new Vector<DoubleMatrix>();
		for (int i = 0; i < (num_layers - 1); i++) {
			DoubleMatrix currentTheta = Theta.get(i);
			DoubleMatrix modTheta = new DoubleMatrix().copy(currentTheta);
			modTheta.putColumn(0, DoubleMatrix.zeros(currentTheta.getRows(), 1));
			gradList.add(DELTA.get(i).div(m).add(modTheta.mul(lambda / m)));
		}

		DoubleMatrix gradiants = new DoubleMatrix().copy(NeuralNetwork.reshapeToVector(gradList));
		return new Tuple<Double, DoubleMatrix>(new Double(J), gradiants);
	}

	private double getSparsityCost(DoubleMatrix X, Vector<DoubleMatrix> Theta, int jCount) {
		DoubleMatrix firstLevelActivations = AutoEncoder.forwardPropPredictFirstActivationOnly(Theta, X);
		double cost = 0d;

		for (int j = 0; j < jCount; j++) {
			cost = cost + this.sparsityParameter / getPj(firstLevelActivations, jCount, j)
					+ (1 - this.sparsityParameter)
					* Math.log((1 - sparsityParameter) / (1 - getPj(firstLevelActivations, jCount, j)));
		}
		return cost * sparsityBeta;
	}

	private DoubleMatrix getSparsityGradientTerm(DoubleMatrix X, Vector<DoubleMatrix> Theta) {
		int jCount = getHiddenNeuronsCount();
		DoubleMatrix ones = DoubleMatrix.ones(X.rows, jCount + 1);
		DoubleMatrix pMatrix = ones.mul(-sparsityParameter);
		DoubleMatrix firstTerm = pMatrix.div(getPis(X, Theta, jCount));

		DoubleMatrix top = DoubleMatrix.ones(X.rows, jCount + 1).sub(pMatrix);
		DoubleMatrix bottom = DoubleMatrix.ones(X.rows, jCount + 1).sub(getPis(X, Theta, jCount));

		DoubleMatrix secondTerm = top.div(bottom);
		DoubleMatrix result = (firstTerm.add(secondTerm)).mul(sparsityBeta);

		double[][] result1 = new double[result.getRows()][result.getColumns()];
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getColumns(); j++) {
				if (i == 0) {
					result1[i][j] = 0d;
				} else {
					result1[i][j] = result.get(i, j);
				}

			}
		}
		return new DoubleMatrix(result1);

	}

	private DoubleMatrix getPis(DoubleMatrix X, Vector<DoubleMatrix> Theta, int jCount) {
		DoubleMatrix firstLevelActivations = AutoEncoder.forwardPropPredictFirstActivationOnly(Theta, X);
		double[][] pjs = new double[X.getRows()][jCount + 1];
		for (int i2 = 0; i2 < X.getRows(); i2++) {
			pjs[i2][0] = 1;
			for (int j = 1; j < jCount + 1; j++) {
				double pj = 0;
				for (int i = 0; i < X.getRows(); i++) {
					pj = pj + firstLevelActivations.get(i, j - 1);
				}
				pj = pj / X.getRows();
				pjs[i2][j] = pj;
			}
		}

		DoubleMatrix m = new DoubleMatrix(pjs);
		return m;
	}

	private double getPj(DoubleMatrix firstLevelActivations, int jCount, int j) {

		double pj = 0;
		for (int i = 0; i < X.getRows(); i++) {
			pj = pj + firstLevelActivations.get(i, j);
		}
		pj = pj / X.getRows();

		return pj;
	}

}
