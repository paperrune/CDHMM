#include <fstream>
#include <iostream>
#include <omp.h>
#include <time.h>

#include "../CHMM.h"

void Read_MNIST(string training_set_images, string training_set_labels, string test_set_images, string test_set_labels, int number_training, int number_test, int *label, double **input) {
	ifstream file(training_set_images, ifstream::binary);

	if (file.is_open()) {
		for (int h = 0, value; h < 4; h++) {
			file.read((char*)(&value), sizeof(int));
		}
		for (int h = 0; h < number_training; h++) {
			unsigned char pixel;

			for (int j = 0; j < 28 * 28; j++) {
				file.read((char*)(&pixel), 1);
				input[h][j] = pixel / 255.0;
			}
		}
		file.close();
	}
	else {
		cerr << "[Read_MNIST], " + training_set_images + " not found" << endl;
	}

	file.open(training_set_labels, ifstream::binary);

	if (file.is_open()) {
		for (int h = 0, value; h < 2; h++) {
			file.read((char*)(&value), sizeof(int));
		}
		for (int h = 0; h < number_training; h++) {
			unsigned char value;

			file.read((char*)(&value), 1);
			label[h] = value;
		}
		file.close();
	}
	else {
		cerr << "[Read_MNIST], " + training_set_labels + " not found" << endl;
	}

	file.open(test_set_images, ifstream::binary);

	if (file.is_open()) {
		for (int h = 0, value; h < 4; h++) {
			file.read((char*)(&value), sizeof(int));
		}
		for (int h = number_training; h < number_training + number_test; h++) {
			unsigned char pixel;

			for (int j = 0; j < 28 * 28; j++) {
				file.read((char*)(&pixel), 1);
				input[h][j] = pixel / 255.0;
			}
		}
		file.close();
	}
	else {
		cerr << "[Read_MNIST], " + test_set_images + " not found" << endl;
	}

	file.open(test_set_labels, ifstream::binary);

	if (file.is_open()) {
		for (int h = 0, value; h < 2; h++) {
			file.read((char*)(&value), sizeof(int));
		}
		for (int h = number_training; h < number_training + number_test; h++) {
			unsigned char value;

			file.read((char*)(&value), 1);
			label[h] = value;
		}
		file.close();
	}
	else {
		cerr << "[Read_MNIST], " + test_set_labels + " not found" << endl;
	}
}

int main() {
	int dimension_event = 56;
	int length_data		= 14;
	int number_iterations = 50;
	int number_threads	= 4;
	int number_training = 60000;
	int number_test		= 10000;

	int number_gaussian_components = 1;

	int *label			= new int[number_training + number_test];
	int *length_event	= new int[number_training + number_test];

	double minimum_variance = 0.05; // prevents overfitting in the case of diagonal covariance

	double **_event = new double*[number_training + number_test];

	vector<string> state_label	= { "0", "0", "0", "1", "1", "1", "2", "2", "2", "3", "3", "3", "4", "4", "4", "5", "5", "5", "6", "6", "6", "7", "7", "7", "8", "8", "8", "9", "9", "9", "", "", "" };
	string type_covariance		= "diagonal";	// <-> "full"
	string type_model			= "bakis";		// <-> "ergodic"

	Continuous_Hidden_Markov_Model *CHMM;

	for (int i = 0; i < number_training + number_test; i++) {
		_event[i] = new double[(length_event[i] = length_data) * dimension_event];
	}
	Read_MNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", number_training, number_test, label, _event);
	omp_set_num_threads(number_threads);

	// Construct CHMM
	{
		int number_states = static_cast<int>(state_label.size());

		set<int> *state_connection = new set<int>[number_states];

		for (int i = 0; i < number_states; i++) {
			for (int j = 0; j < number_states; j++) {
				// for ergodic model
				// state_connection[i].insert(j);

				// for bakis model
				if (i == j || (i % 3 == 2 && j % 3 == 0) || j - i == 1) {
					state_connection[i].insert(j);
				}
			}
		}
		CHMM = new Continuous_Hidden_Markov_Model(state_connection, type_covariance, type_model, state_label, dimension_event, number_gaussian_components, number_states);

		delete[] state_connection;
	}

	// Initialize and Train CHMM
	{
		vector<int> *state = new vector<int>[number_training];

		for (int i = 0; i < number_training; i++) {
			for (int j = 0; j < 9; j++) {
				state[i].push_back(((j / 3 == 0 || j / 3 == 2) ? (30) : (3 * label[i])) + (j % 3));
			}
		}

		CHMM->Initialize(number_training, length_event, _event);

		for (int h = 0, time = clock(); h < number_iterations; h++) {
			int number_correct[2] = { 0, };

			double log_likelihood = CHMM->Baum_Welch_Algorithm(number_training, length_event, state, minimum_variance, _event);

			#pragma omp parallel for
			for (int i = 0; i < number_training + number_test; i++) {
				int *optimal_state_sequence;

				string optimal_label_sequence;

				CHMM->Viterbi_Algorithm(&optimal_label_sequence, &optimal_state_sequence, length_event[i], _event[i]);

				/*#pragma omp critical
				{
					for (int t = 0; t < length_event[i]; t++) {
						cout << optimal_state_sequence[t] << ' ';
					}
					cout << endl;
				}*/

				// cout << optimal_label_sequence << endl;

				#pragma omp atomic
				number_correct[(i < number_training) ? (0) : (1)] += (atoi(&optimal_label_sequence[0]) == label[i]);

				if (optimal_state_sequence) {
					delete[] optimal_state_sequence;
				}
			}
			// printf(".");	CHMM->Save_Model("CHMM.txt");

			printf("score: %d / %d, %d / %d  L: %lf  step %d  %.2lf sec\n", number_correct[0], number_training, number_correct[1], number_test, log_likelihood, h + 1, (double)(clock() - time) / CLOCKS_PER_SEC);
		}
		delete[] state;
	}

	for (int i = 0; i < number_training + number_test; i++) {
		delete[] _event[i];
	}
	delete[] _event;
	delete[] label;
	delete[] length_event;
	delete CHMM;

	return 0;
}
